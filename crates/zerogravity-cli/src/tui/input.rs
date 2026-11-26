//! Visible hit targets only. Pointer translation does not own runtime actions.
use super::{
    controls::Bound,
    layout::{Motion, Pane, Scrollbar, Splitter},
    library::{Key, Kind, Shelf},
    view::View,
    work::Owner,
};
use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::layout::{Position, Rect};
use zerogravity::configuration::ThemeMode;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Target {
    Pane(Pane),
    Navigate(View),
    Navigation,
    Row(Kind, Key),
    RowMenu(Kind, Key),
    Owner(Owner),
    OwnerMenu(Owner),
    Log(u64),
    LogDetails(u64),
    Shelf(Shelf),
    Split(Splitter),
    Scrollbar(Scrollbar),
    Maximize(Pane),
    Focus(Pane),
    Theme(ThemeMode),
    Help,
    More,
    SessionMenu,
    ResponseTab,
    SearchFocus,
    SearchClear,
    Command(Bound),
    AcceptModal,
    DismissModal,
    PromptMode,
    Choice(usize),
    ChoiceYaml,
    EditorRow { start: usize, end: usize },
    EditorBar(Scrollbar),
    EditorViewport,
    ModalScroll(u16),
}
#[derive(Debug, Default)]
pub(super) struct Hits(Vec<(Rect, Target)>);
impl Hits {
    pub fn clear(&mut self) {
        self.0.clear();
    }
    pub fn add(&mut self, rect: Rect, target: Target) {
        if !rect.is_empty() {
            self.0.push((rect, target));
        }
    }
    fn at(&self, pos: Position) -> Option<&Target> {
        self.0
            .iter()
            .rev()
            .find(|(rect, _)| rect.contains(pos))
            .map(|(_, target)| target)
    }
    fn editor_point(&self, pos: Position, extend: bool) -> Option<Intent> {
        self.0.iter().rev().find_map(|(rect, target)| {
            if let Target::EditorRow { start, end } = target
                && rect.contains(pos)
            {
                Some(Intent::EditPoint {
                    start: *start,
                    end: *end,
                    column: pos.x - rect.x,
                    extend,
                })
            } else {
                None
            }
        })
    }
    /// Only used during an existing editor capture. Motion outside
    /// the field extends the selection, never becomes a button press.
    fn editor_drag(&self, pos: Position) -> Option<Intent> {
        if let Some(intent) = self.editor_point(pos, true) {
            return Some(intent);
        }
        let rows = self.0.iter().filter_map(|(rect, target)| {
            if let Target::EditorRow { start, end } = target {
                Some((*rect, *start, *end))
            } else {
                None
            }
        });
        let mut rows = rows.peekable();
        let first = *rows.peek()?;
        let last = rows.clone().last().unwrap_or(first);
        if pos.y < first.0.y || pos.y >= last.0.bottom() {
            let edge = if pos.y < first.0.y { -1 } else { 1 };
            return Some(Intent::EditEdge(
                edge,
                pos.x.saturating_sub(first.0.x).min(first.0.width),
            ));
        }
        rows.find(|(rect, _, _)| pos.y >= rect.y && pos.y < rect.bottom())
            .map(|(rect, start, end)| Intent::EditPoint {
                start,
                end,
                column: pos.x.saturating_sub(rect.x).min(rect.width),
                extend: true,
            })
    }
    fn pane(&self, pos: Position) -> Option<Pane> {
        self.0.iter().rev().find_map(|(rect, target)| {
            if let Target::Pane(pane) = target
                && rect.contains(pos)
            {
                Some(*pane)
            } else {
                None
            }
        })
    }
}
#[derive(Debug)]
enum Capture {
    Press(Target),
    Context(Target),
    Split(Splitter),
    Scroll(Scrollbar, u16),
    Editor,
    EditorBar(Scrollbar, u16),
}
#[derive(Debug, Default)]
pub(super) struct Pointer(Option<Capture>);
pub(super) enum Intent {
    Click(Target),
    Context(Target),
    Resize(Splitter, u16, u16),
    Scroll(Pane, Motion),
    ScrollModal(i16, u16),
    EditPoint {
        start: usize,
        end: usize,
        column: u16,
        extend: bool,
    },
    EditScroll(Motion),
    EditEdge(i16, u16),
}
impl Pointer {
    pub fn cancel(&mut self) {
        self.0 = None;
    }
    pub fn invalidate_selection(&mut self) {
        if matches!(
            self.0,
            Some(Capture::Press(
                Target::Row(..)
                    | Target::RowMenu(..)
                    | Target::Owner(_)
                    | Target::OwnerMenu(_)
                    | Target::Log(_)
                    | Target::LogDetails(_)
                    | Target::Command(_)
            )) | Some(Capture::Context(_))
        ) {
            self.cancel();
        }
    }
    pub fn event(&mut self, event: MouseEvent, hits: &Hits) -> Option<Intent> {
        let pos = Position::new(event.column, event.row);
        match event.kind {
            MouseEventKind::Down(MouseButton::Right) => {
                self.cancel();
                if let Some(
                    target @ (Target::Row(..)
                    | Target::RowMenu(..)
                    | Target::Owner(_)
                    | Target::OwnerMenu(_)
                    | Target::Log(_)
                    | Target::LogDetails(_)),
                ) = hits.at(pos)
                {
                    self.0 = Some(Capture::Context(target.clone()));
                }
                None
            }
            MouseEventKind::Drag(MouseButton::Right) => {
                self.cancel();
                None
            }
            MouseEventKind::Up(MouseButton::Right) => match self.0.take()? {
                Capture::Context(target) if hits.at(pos) == Some(&target) => {
                    Some(Intent::Context(target))
                }
                _ => None,
            },
            MouseEventKind::Down(MouseButton::Left) => {
                self.cancel();
                match hits.at(pos)?.clone() {
                    Target::Split(split) => self.0 = Some(Capture::Split(split)),
                    Target::EditorRow { .. } => {
                        self.0 = Some(Capture::Editor);
                        return hits.editor_point(pos, false);
                    }
                    Target::EditorBar(bar) => {
                        let grab = if bar.thumb.contains(pos) {
                            event.row - bar.thumb.y
                        } else {
                            bar.thumb.height / 2
                        };
                        self.0 = Some(Capture::EditorBar(bar, grab));
                        return Some(Intent::EditScroll(Motion::At(bar.offset(event.row, grab))));
                    }
                    Target::Scrollbar(bar) => {
                        let grab = if bar.thumb.contains(pos) {
                            event.row - bar.thumb.y
                        } else {
                            bar.thumb.height / 2
                        };
                        self.0 = Some(Capture::Scroll(bar, grab));
                        return Some(Intent::Scroll(
                            bar.pane,
                            Motion::At(bar.offset(event.row, grab)),
                        ));
                    }
                    target => self.0 = Some(Capture::Press(target)),
                }
                None
            }
            MouseEventKind::Drag(MouseButton::Left) => match &self.0 {
                Some(Capture::Split(split)) => {
                    Some(Intent::Resize(*split, event.column, event.row))
                }
                Some(Capture::Scroll(bar, grab)) => Some(Intent::Scroll(
                    bar.pane,
                    Motion::At(bar.offset(event.row, *grab)),
                )),
                Some(Capture::Editor) => hits.editor_drag(pos),
                Some(Capture::EditorBar(bar, grab)) => {
                    Some(Intent::EditScroll(Motion::At(bar.offset(event.row, *grab))))
                }
                _ => {
                    self.cancel();
                    None
                }
            },
            MouseEventKind::Up(MouseButton::Left) => match self.0.take()? {
                Capture::Press(target) if hits.at(pos) == Some(&target) => {
                    Some(Intent::Click(target))
                }
                _ => None,
            },
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                self.cancel();
                let delta = if event.kind == MouseEventKind::ScrollUp {
                    -3
                } else {
                    3
                };
                if hits
                    .0
                    .iter()
                    .any(|(rect, target)| *target == Target::EditorViewport && rect.contains(pos))
                {
                    return Some(Intent::EditScroll(Motion::Step(delta)));
                }
                if let Some(max) = hits.0.iter().rev().find_map(|(rect, target)| {
                    if let Target::ModalScroll(max) = target
                        && rect.contains(pos)
                    {
                        Some(*max)
                    } else {
                        None
                    }
                }) {
                    Some(Intent::ScrollModal(delta, max))
                } else {
                    hits.pane(pos)
                        .map(|pane| Intent::Scroll(pane, Motion::Step(delta)))
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyModifiers;
    fn event(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }
    #[test]
    fn captured_editor_drags_past_edges_without_click_through() {
        let mut hits = Hits::default();
        let mut pointer = Pointer::default();
        hits.add(
            Rect::new(2, 3, 8, 1),
            Target::EditorRow { start: 0, end: 4 },
        );
        hits.add(
            Rect::new(2, 4, 8, 1),
            Target::EditorRow { start: 5, end: 9 },
        );
        hits.add(Rect::new(2, 5, 8, 1), Target::AcceptModal);
        assert!(matches!(
            pointer.event(event(MouseEventKind::Down(MouseButton::Left), 2, 3), &hits),
            Some(Intent::EditPoint { extend: false, .. })
        ));
        assert!(matches!(
            pointer.event(event(MouseEventKind::Drag(MouseButton::Left), 12, 3), &hits),
            Some(Intent::EditPoint {
                column: 8,
                extend: true,
                ..
            })
        ));
        assert!(matches!(
            pointer.event(event(MouseEventKind::Drag(MouseButton::Left), 5, 5), &hits),
            Some(Intent::EditEdge(1, 3))
        ));
        assert!(
            pointer
                .event(event(MouseEventKind::Up(MouseButton::Left), 5, 5), &hits)
                .is_none()
        );
    }
    #[test]
    fn right_click_is_exact_visible_context_only() {
        let mut hits = Hits::default();
        let mut pointer = Pointer::default();
        let a = Target::Row(Kind::Models, Key::Snapshot("a".into()));
        let b = Target::Row(Kind::Models, Key::Snapshot("b".into()));
        hits.add(Rect::new(0, 0, 10, 1), a.clone());
        hits.add(Rect::new(0, 1, 10, 1), b.clone());
        assert!(
            pointer
                .event(event(MouseEventKind::Down(MouseButton::Right), 1, 0), &hits)
                .is_none()
        );
        assert!(
            pointer
                .event(event(MouseEventKind::Up(MouseButton::Right), 1, 1), &hits)
                .is_none()
        );
        pointer.event(event(MouseEventKind::Down(MouseButton::Right), 1, 0), &hits);
        assert!(
            matches!(pointer.event(event(MouseEventKind::Up(MouseButton::Right),1,0),&hits),Some(Intent::Context(target)) if target==a)
        );
        pointer.event(event(MouseEventKind::Down(MouseButton::Right), 1, 0), &hits);
        pointer.invalidate_selection();
        assert!(
            pointer
                .event(event(MouseEventKind::Up(MouseButton::Right), 1, 0), &hits)
                .is_none()
        );
    }
    #[test]
    fn only_same_visible_target_can_click_and_wheel_does_not_select() {
        let mut hits = Hits::default();
        let mut pointer = Pointer::default();
        let area = Rect::new(0, 0, 10, 10);
        hits.add(area, Target::Pane(Pane::Main));
        hits.add(Rect::new(1, 1, 8, 1), Target::Navigate(View::Models));
        assert!(
            pointer
                .event(event(MouseEventKind::Down(MouseButton::Left), 2, 1), &hits)
                .is_none()
        );
        assert!(
            pointer
                .event(event(MouseEventKind::Up(MouseButton::Left), 2, 2), &hits)
                .is_none()
        );
        pointer.event(event(MouseEventKind::Down(MouseButton::Left), 2, 1), &hits);
        assert!(matches!(
            pointer.event(event(MouseEventKind::Up(MouseButton::Left), 2, 1), &hits),
            Some(Intent::Click(Target::Navigate(View::Models)))
        ));
        assert!(matches!(
            pointer.event(event(MouseEventKind::ScrollDown, 2, 1), &hits),
            Some(Intent::Scroll(Pane::Main, Motion::Step(3)))
        ));
        pointer.event(event(MouseEventKind::Down(MouseButton::Left), 2, 1), &hits);
        hits.clear();
        assert!(
            pointer
                .event(event(MouseEventKind::Up(MouseButton::Left), 2, 1), &hits)
                .is_none()
        );
    }
}
