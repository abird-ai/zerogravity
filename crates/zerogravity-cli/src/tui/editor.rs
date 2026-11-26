//! Bounded native text draft: one UTF-8 value, a grapheme caret and selection,
//! and a client-local visual-row viewport. No file, clipboard or submit owner.
use super::layout::{Motion, Viewport};
use std::ops::Range;
use unicode_segmentation::{GraphemeCursor, UnicodeSegmentation};
use unicode_width::UnicodeWidthStr;

fn previous(value: &str, at: usize) -> usize {
    GraphemeCursor::new(at, value.len(), true)
        .prev_boundary(value, 0)
        .expect("complete text")
        .unwrap_or(0)
}
fn next(value: &str, at: usize) -> usize {
    GraphemeCursor::new(at, value.len(), true)
        .next_boundary(value, 0)
        .expect("complete text")
        .unwrap_or(value.len())
}
fn boundary(value: &str, at: usize) -> usize {
    let mut cursor = GraphemeCursor::new(at, value.len(), true);
    if cursor.is_boundary(value, 0).expect("complete text") {
        at
    } else {
        next(value, at)
    }
}

/// The same sanitized display and width calculation is used for painting,
/// row wrapping and hit translation. Tabs are four visible cells; the draft
/// retains the original tab. Control characters are never emitted.
pub(super) fn display(grapheme: &str, width: u16) -> &str {
    if grapheme == "\t" {
        if width >= 4 { "    " } else { " " }
    } else if grapheme.chars().any(char::is_control)
        || UnicodeWidthStr::width(grapheme) > usize::from(width)
    {
        "�"
    } else {
        grapheme
    }
}

#[derive(Debug)]
pub(super) struct VisualRows {
    pub rows: Vec<Range<usize>>,
    width: u16,
}
impl VisualRows {
    pub fn new(value: &str, width: u16) -> Self {
        let width = width.max(1);
        let mut rows = Vec::new();
        let mut start = 0;
        let mut column = 0;
        for (at, grapheme) in value.grapheme_indices(true) {
            if grapheme.ends_with('\n') {
                rows.push(start..at);
                start = at + grapheme.len();
                column = 0;
                continue;
            }
            let cells = UnicodeWidthStr::width(display(grapheme, width));
            if column != 0 && column + cells > usize::from(width) {
                rows.push(start..at);
                start = at;
                column = 0;
            }
            column += cells;
        }
        rows.push(start..value.len());
        Self { rows, width }
    }
    pub fn position(&self, value: &str, caret: usize) -> (usize, u16) {
        let row = self
            .rows
            .partition_point(|range| range.start <= caret)
            .saturating_sub(1);
        let range = &self.rows[row];
        let column: usize = value[range.start..caret.min(range.end)]
            .graphemes(true)
            .map(|g| UnicodeWidthStr::width(display(g, self.width)))
            .sum();
        (row, column.min(usize::from(self.width)) as u16)
    }
}
pub(super) fn hit(value: &str, range: Range<usize>, width: u16, column: u16) -> usize {
    let Some(line) = value.get(range.clone()) else {
        return value.len();
    };
    let mut x = 0;
    for (offset, grapheme) in line.grapheme_indices(true) {
        let cells = UnicodeWidthStr::width(display(grapheme, width));
        if usize::from(column) < x + cells {
            // The right half of a wide grapheme places the caret after it.
            return range.start
                + offset
                + if usize::from(column) - x >= cells.div_ceil(2) {
                    grapheme.len()
                } else {
                    0
                };
        }
        x += cells;
    }
    range.end
}

#[derive(Debug)]
pub(super) struct Editor {
    pub caret: usize,
    anchor: Option<usize>,
    pub viewport: Viewport,
    width: u16,
    reveal: bool,
}
impl Editor {
    pub fn new(value: &str, caret: usize) -> Self {
        Self {
            caret: boundary(value, caret),
            anchor: None,
            viewport: Viewport::default(),
            width: 1,
            reveal: true,
        }
    }
    pub fn selection(&self) -> Range<usize> {
        let anchor = self.anchor.unwrap_or(self.caret);
        anchor.min(self.caret)..anchor.max(self.caret)
    }
    pub fn point(&mut self, value: &str, range: Range<usize>, column: u16, extend: bool) {
        self.move_to(value, hit(value, range, self.width, column), extend);
        // Pointer is already in the visible viewport. Do not shift it when
        // the user selects the end of a wrapped row.
        self.reveal = false;
    }
    /// One visual-row step per real drag event past the top/bottom.
    /// The original anchor stays fixed. No timers or synthetic input.
    pub fn drag_edge(&mut self, value: &str, delta: i16, column: u16) {
        let rows = VisualRows::new(value, self.width);
        self.viewport
            .measure(rows.rows.len(), self.viewport.page.max(1));
        self.viewport.scroll(Motion::Step(delta));
        let index = if delta < 0 {
            self.viewport.offset
        } else {
            self.viewport
                .offset
                .saturating_add(self.viewport.page.max(1) as usize - 1)
                .min(rows.rows.len() - 1)
        };
        self.point(value, rows.rows[index].clone(), column, true);
    }
    fn move_to(&mut self, value: &str, at: usize, extend: bool) {
        if extend {
            self.anchor.get_or_insert(self.caret);
        } else {
            self.anchor = None;
        }
        self.caret = boundary(value, at);
        self.reveal = true;
    }
    pub fn move_horizontal(&mut self, value: &str, right: bool, extend: bool) {
        let selected = self.selection();
        let at = if !extend && !selected.is_empty() {
            if right { selected.end } else { selected.start }
        } else if right {
            next(value, self.caret)
        } else {
            previous(value, self.caret)
        };
        self.move_to(value, at, extend);
    }
    pub fn move_vertical(&mut self, value: &str, down: bool, extend: bool) {
        let rows = VisualRows::new(value, self.width);
        let (row, column) = rows.position(value, self.caret);
        let index = if down {
            (row + 1).min(rows.rows.len() - 1)
        } else {
            row.saturating_sub(1)
        };
        if index != row {
            self.move_to(
                value,
                hit(value, rows.rows[index].clone(), self.width, column),
                extend,
            );
        }
    }
    pub fn edge(&mut self, value: &str, end: bool, whole: bool, extend: bool) {
        let at = match (whole, end) {
            (true, true) => value.len(),
            (true, false) => 0,
            (false, true) => value[self.caret..].find('\n').map_or(value.len(), |i| {
                let at = self.caret + i;
                if value[..at].ends_with('\r') {
                    at - 1
                } else {
                    at
                }
            }),
            (false, false) => value[..self.caret].rfind('\n').map_or(0, |i| i + 1),
        };
        self.move_to(value, at, extend);
    }
    pub fn replace(&mut self, value: &mut String, insertion: &str, limit: usize) -> bool {
        let range = self.selection();
        if value
            .len()
            .saturating_sub(range.len())
            .saturating_add(insertion.len())
            > limit
        {
            return false;
        }
        let at = range.start + insertion.len();
        value.replace_range(range, insertion);
        self.move_to(value, at, false);
        true
    }
    pub fn delete(&mut self, value: &mut String, backward: bool) {
        if self.selection().is_empty() {
            self.anchor = Some(if backward {
                previous(value, self.caret)
            } else {
                next(value, self.caret)
            });
        }
        let _ = self.replace(value, "", usize::MAX);
    }
    pub fn clear(&mut self, value: &mut String) {
        value.clear();
        self.move_to(value, 0, false);
    }
    pub fn measure(&mut self, value: &str, width: u16, height: u16) -> VisualRows {
        let width = width.max(1);
        let anchor = if self.width != width && !self.reveal {
            VisualRows::new(value, self.width)
                .rows
                .get(self.viewport.offset)
                .map(|r| r.start)
        } else {
            None
        };
        self.width = width;
        let rows = VisualRows::new(value, self.width);
        if let Some(anchor) = anchor {
            self.viewport.offset = rows.position(value, anchor).0;
        }
        self.viewport.measure(rows.rows.len(), height);
        if std::mem::take(&mut self.reveal) {
            self.viewport.reveal(rows.position(value, self.caret).0);
        }
        rows
    }
    pub fn scroll(&mut self, motion: Motion) {
        self.reveal = false;
        self.viewport.scroll(motion);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn edge_drag_scrolls_and_preserves_selection_anchor() {
        let value = "first\nsecond\nthird\nfourth";
        let mut editor = Editor::new(value, 0);
        let visual = editor.measure(value, 10, 2);
        editor.point(value, visual.rows[0].clone(), 0, false);
        editor.drag_edge(value, 1, 99);
        assert_eq!(editor.viewport.offset, 1);
        assert_eq!(&value[editor.selection()], "first\nsecond\nthird");
        editor.drag_edge(value, 1, 99);
        assert_eq!(editor.viewport.offset, 2);
        assert_eq!(&value[editor.selection()], value);
        editor.drag_edge(value, -1, 0);
        assert_eq!(&value[editor.selection()], "first\n");
    }
    #[test]
    fn graphemes_mouse_selection_wrap_and_bounded_replace() {
        let mut value = "a界e\u{301}z\nlong".to_string();
        let mut editor = Editor::new(&value, 0);
        let rows = editor.measure(&value, 4, 2);
        assert_eq!(&value[rows.rows[0].clone()], "a界e\u{301}");
        editor.point(&value, rows.rows[0].clone(), 1, false);
        editor.point(&value, rows.rows[0].clone(), 3, true);
        assert_eq!(&value[editor.selection()], "界");
        assert!(!editor.replace(&mut value, "abcdef", 4));
        assert_eq!(&value[editor.selection()], "界");
        assert!(editor.replace(&mut value, "X", 32));
        assert_eq!(value, "aXe\u{301}z\nlong");
        editor.move_horizontal(&value, true, false);
        editor.delete(&mut value, true);
        assert_eq!(value, "aXz\nlong");
        editor.move_horizontal(&value, true, false);
        editor.edge(&value, true, true, false);
        editor.measure(&value, 3, 1);
        assert_eq!(editor.viewport.offset, editor.viewport.max);
        editor.scroll(Motion::Start);
        editor.measure(&value, 3, 1);
        assert_eq!(editor.viewport.offset, 0);
        assert_eq!(editor.caret, value.len());
    }
    #[test]
    fn long_draft_can_reach_end_beyond_paragraph_scroll_limit() {
        let value = "\n".repeat(70_000);
        let mut editor = Editor::new(&value, value.len());
        let rows = editor.measure(&value, 20, 10);
        assert_eq!(rows.rows.len(), 70_001);
        assert_eq!(editor.viewport.offset, 69_991);
        editor.scroll(Motion::Start);
        editor.measure(&value, 15, 10);
        assert_eq!(editor.viewport.offset, 0);
    }
    #[test]
    fn insertion_join_and_crlf_preserve_valid_caret() {
        let mut value = "a\r\nb".to_owned();
        let mut editor = Editor::new(&value, 1);
        assert!(editor.replace(&mut value, "\u{301}", 64));
        assert_eq!(editor.caret, 3);
        editor.move_horizontal(&value, true, false);
        assert_eq!(editor.caret, 5);
        editor.delete(&mut value, true);
        assert_eq!(value, "a\u{301}b");
        editor.delete(&mut value, true);
        assert_eq!(value, "b");
    }
}
