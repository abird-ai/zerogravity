//! Client-local cell geometry and scroll state. Preferred sizes survive clamps.
use ratatui::layout::Rect;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Pane {
    Rail,
    #[default]
    Main,
    Detail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SplitKind {
    Rail,
    Horizontal,
    Vertical,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Splitter {
    pub kind: SplitKind,
    pub parent: Rect,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Motion {
    Step(i16),
    Page(i16),
    Start,
    End,
    At(usize),
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Viewport {
    pub offset: usize,
    pub max: usize,
    pub page: u16,
}
impl Viewport {
    pub fn measure(&mut self, rows: usize, height: u16) {
        self.page = height;
        self.max = rows.saturating_sub(height as usize);
        self.offset = self.offset.min(self.max);
    }
    pub fn reveal(&mut self, row: usize) {
        if row < self.offset {
            self.offset = row;
        } else if row >= self.offset.saturating_add(self.page as usize) {
            self.offset = row
                .saturating_add(1)
                .saturating_sub(self.page.max(1) as usize);
        }
        self.offset = self.offset.min(self.max);
    }
    pub fn scroll(&mut self, motion: Motion) {
        self.offset = match motion {
            Motion::Step(delta) => self.offset.saturating_add_signed(delta as isize),
            Motion::Page(delta) => self.offset.saturating_add_signed(
                isize::from(delta).saturating_mul(self.page.saturating_sub(1).max(1) as isize),
            ),
            Motion::Start => 0,
            Motion::End => self.max,
            Motion::At(offset) => offset,
        }
        .min(self.max);
    }
}
/// Remember a visible item, not an index into a mutable observation.
/// Keep the preferred top through a temporary clamp or empty refresh.
/// Explicit scroll/reveal supersedes it; an evicted key falls back safely.
#[derive(Debug)]
pub(super) struct Anchor<K> {
    key: Option<(K, usize)>,
    preferred: usize,
    painted: Option<usize>,
}
impl<K> Default for Anchor<K> {
    fn default() -> Self {
        Self {
            key: None,
            preferred: 0,
            painted: None,
        }
    }
}
impl<K> Anchor<K> {
    pub fn last(&self) -> Option<usize> {
        self.painted
    }

    pub fn restore(
        &mut self,
        offset: usize,
        reveal: bool,
        empty: bool,
        find: impl FnOnce(&K) -> Option<usize>,
        missing: usize,
    ) -> usize {
        if reveal || self.painted != Some(offset) {
            self.key = None;
            self.preferred = offset;
        } else if !empty && let Some((key, gap)) = &self.key {
            if let Some(row) = find(key) {
                self.preferred = row.saturating_sub(*gap);
            } else {
                self.key = None;
                self.preferred = missing;
            }
        }
        self.preferred
    }

    pub fn remember(&mut self, offset: usize, next: impl FnOnce() -> Option<(usize, K)>) {
        if self.key.is_none() {
            self.preferred = offset;
            self.key = next().map(|(row, key)| (key, row.saturating_sub(offset)));
        }
        self.painted = Some(offset);
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Scrollbar {
    pub pane: Pane,
    pub track: Rect,
    pub thumb: Rect,
    pub max: usize,
}
impl Scrollbar {
    pub fn new(pane: Pane, track: Rect, viewport: Viewport) -> Option<Self> {
        if track.height < 2 || viewport.max == 0 {
            return None;
        }
        let total = viewport.max.saturating_add(viewport.page as usize);
        let height = ((usize::from(track.height) * usize::from(viewport.page) / total) as u16)
            .clamp(1, track.height - 1);
        let top = (usize::from(track.height - height).saturating_mul(viewport.offset)
            / viewport.max) as u16;
        Some(Self {
            pane,
            track,
            thumb: Rect::new(track.x, track.y + top, 1, height),
            max: viewport.max,
        })
    }
    pub fn offset(self, row: u16, grab: u16) -> usize {
        let range = self.track.height - self.thumb.height;
        let position = row
            .saturating_sub(self.track.y)
            .saturating_sub(grab)
            .min(range);
        usize::from(position).saturating_mul(self.max) / usize::from(range)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ViewState {
    main: Viewport,
    detail: Viewport,
    output: Viewport,
    maximized: Option<Pane>,
    follow_output: bool,
    reveal: bool,
    focus: Pane,
}
#[derive(Debug)]
pub(super) struct Workspace {
    pub focus: Pane,
    pub has_detail: bool,
    rail_width: Option<u16>,
    horizontal: Option<u16>,
    vertical: u16,
    rail: Viewport,
    views: [ViewState; 9],
}
impl Default for Workspace {
    fn default() -> Self {
        Self {
            focus: Pane::Main,
            has_detail: false,
            rail_width: None,
            horizontal: None,
            vertical: 500,
            rail: Viewport::default(),
            views: [ViewState {
                follow_output: true,
                reveal: true,
                ..Default::default()
            }; 9],
        }
    }
}
impl Workspace {
    pub fn viewport(&mut self, view: usize, pane: Pane) -> &mut Viewport {
        match pane {
            Pane::Rail => &mut self.rail,
            Pane::Main => &mut self.views[view].main,
            Pane::Detail => &mut self.views[view].detail,
        }
    }
    pub fn output(&mut self, view: usize) -> &mut Viewport {
        &mut self.views[view].output
    }
    pub fn focus(&mut self, view: usize, pane: Pane) {
        self.focus = pane;
        if pane != Pane::Rail {
            self.views[view].focus = pane;
            if self.views[view].maximized.is_some() {
                self.views[view].maximized = Some(pane);
            }
        }
    }
    pub fn toggle_maximize(&mut self, view: usize, pane: Pane) {
        if pane == Pane::Rail || !self.has_detail {
            return;
        }
        self.focus = pane;
        self.views[view].focus = pane;
        let maximized = &mut self.views[view].maximized;
        *maximized = if *maximized == Some(pane) {
            None
        } else {
            Some(pane)
        };
    }
    pub fn preferred_focus(&self, view: usize) -> Pane {
        self.views[view].maximized.unwrap_or(self.views[view].focus)
    }
    pub fn maximized(&self, view: usize, pane: Pane) -> bool {
        self.views[view].maximized == Some(pane)
    }
    pub fn request_reveal(&mut self, view: usize) {
        self.views[view].reveal = true;
    }
    pub fn take_reveal(&mut self, view: usize) -> bool {
        std::mem::take(&mut self.views[view].reveal)
    }
    pub fn follow_output(&self, view: usize) -> bool {
        self.views[view].follow_output
    }
    pub fn set_follow_output(&mut self, view: usize, follow: bool) {
        self.views[view].follow_output = follow;
    }

    /// Rail is full-width when selected in a narrow terminal; otherwise it
    /// collapses without losing the preferred width or cursor.
    pub fn shell(&self, body: Rect) -> (Rect, Rect, Option<(Rect, Splitter)>) {
        if body.width < 70 {
            return if self.focus == Pane::Rail {
                (body, Rect::default(), None)
            } else {
                (Rect::default(), body, None)
            };
        }
        let default = if body.width < 100 { 18 } else { 22 };
        let width = self
            .rail_width
            .unwrap_or(default)
            .clamp(15, 32.min(body.width - 33));
        split(body, width, SplitKind::Rail)
    }
    /// A single split, or one visible pane when the minimums do not fit.
    pub fn inner(
        &self,
        area: Rect,
        view: usize,
        horizontal_default: u16,
    ) -> (Rect, Rect, Option<(Rect, Splitter)>) {
        if let Some(pane) = self.views[view].maximized {
            return single(area, pane);
        }
        if area.width >= 61 && area.height >= 4 {
            let available = area.width - 1;
            let first = portion(available, self.horizontal.unwrap_or(horizontal_default))
                .clamp(32, available - 28);
            split(area, first, SplitKind::Horizontal)
        } else if area.height >= 13 {
            let available = area.height - 1;
            let first = portion(available, self.vertical).clamp(6, available - 6);
            split(area, first, SplitKind::Vertical)
        } else {
            single(area, self.focus)
        }
    }
    pub fn drag(&mut self, splitter: Splitter, column: u16, row: u16) {
        let parent = splitter.parent;
        match splitter.kind {
            SplitKind::Rail => {
                self.rail_width = Some(column.saturating_sub(parent.x).clamp(15, 32));
            }
            SplitKind::Horizontal => {
                let n = parent.width.saturating_sub(1);
                if n >= 60 {
                    let first = column.saturating_sub(parent.x).clamp(32, n - 28);
                    self.horizontal = Some(ratio(first, n));
                }
            }
            SplitKind::Vertical => {
                let n = parent.height.saturating_sub(1);
                if n >= 12 {
                    let first = row.saturating_sub(parent.y).clamp(6, n - 6);
                    self.vertical = ratio(first, n);
                }
            }
        }
    }
    pub fn adjust(&mut self, kind: SplitKind, delta: i16, horizontal_default: u16) {
        match kind {
            SplitKind::Rail => {
                self.rail_width = Some(
                    self.rail_width
                        .unwrap_or(22)
                        .saturating_add_signed(delta)
                        .clamp(15, 32),
                )
            }
            SplitKind::Horizontal => {
                self.horizontal = Some(
                    self.horizontal
                        .unwrap_or(horizontal_default)
                        .saturating_add_signed(delta * 25)
                        .clamp(100, 900),
                )
            }
            SplitKind::Vertical => {
                self.vertical = self
                    .vertical
                    .saturating_add_signed(delta * 50)
                    .clamp(100, 900)
            }
        }
    }
}
fn portion(size: u16, ratio: u16) -> u16 {
    ((u32::from(size) * u32::from(ratio) + 500) / 1000) as u16
}
fn ratio(first: u16, size: u16) -> u16 {
    ((u32::from(first) * 1000 + u32::from(size) / 2) / u32::from(size)) as u16
}
fn single(area: Rect, pane: Pane) -> (Rect, Rect, Option<(Rect, Splitter)>) {
    if pane == Pane::Detail {
        (Rect::default(), area, None)
    } else {
        (area, Rect::default(), None)
    }
}
fn split(area: Rect, first: u16, kind: SplitKind) -> (Rect, Rect, Option<(Rect, Splitter)>) {
    let (a, track, b) = if kind == SplitKind::Vertical {
        (
            Rect::new(area.x, area.y, area.width, first),
            Rect::new(area.x, area.y + first, area.width, 1),
            Rect::new(
                area.x,
                area.y + first + 1,
                area.width,
                area.height - first - 1,
            ),
        )
    } else {
        (
            Rect::new(area.x, area.y, first, area.height),
            Rect::new(area.x + first, area.y, 1, area.height),
            Rect::new(
                area.x + first + 1,
                area.y,
                area.width - first - 1,
                area.height,
            ),
        )
    };
    (a, b, Some((track, Splitter { kind, parent: area })))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preferred_split_survives_collapse_and_maximize() {
        let mut state = Workspace {
            has_detail: true,
            ..Default::default()
        };
        let wide = Rect::new(0, 0, 150, 30);
        let (_, _, rail) = state.shell(wide);
        state.drag(rail.unwrap().1, 29, 0);
        assert_eq!(state.shell(wide).0.width, 29);
        assert_eq!(state.inner(wide, 1, 380).0.width, 57);
        let (_, _, inner) = state.inner(wide, 3, 550);
        state.drag(inner.unwrap().1, 90, 0);
        let preferred = state.inner(wide, 3, 550).0;
        assert_eq!(state.inner(wide, 1, 380).0, preferred);
        state.toggle_maximize(3, Pane::Detail);
        assert_eq!(state.inner(wide, 3, 550).1, wide);
        state.toggle_maximize(3, Pane::Detail);
        assert_eq!(state.inner(wide, 3, 550).0, preferred);
        assert_eq!(state.inner(Rect::new(0, 0, 40, 8), 3, 550).1.width, 40);
        assert_eq!(state.inner(wide, 3, 550).0, preferred);
        assert_eq!(state.shell(Rect::new(0, 0, 50, 8)).1.width, 50);
        assert_eq!(state.shell(wide).0.width, 29);
        let compact = Rect::new(0, 0, 40, 20);
        let (_, _, vertical) = state.inner(compact, 3, 550);
        state.drag(vertical.unwrap().1, 0, 18);
        assert_eq!(state.inner(compact, 3, 550).0.height, 13);
    }
    #[test]
    fn identity_anchor_survives_reorder_clamp_and_empty_refresh() {
        let mut anchor = Anchor::default();
        let mut viewport = Viewport::default();
        viewport.measure(10, 3);
        viewport.scroll(Motion::At(4));
        anchor.restore(viewport.offset, false, false, |_: &u64| None, 0);
        anchor.remember(4, || Some((4, 42)));
        // Reorder: the same observed item is now at row 6.
        assert_eq!(
            anchor.restore(4, false, false, |key| (*key == 42).then_some(6), 0),
            6
        );
        viewport.offset = 6;
        viewport.measure(10, 8);
        assert_eq!(viewport.offset, 2);
        anchor.remember(2, || panic!("clamp is not a new anchor"));
        assert_eq!(anchor.restore(2, false, true, |_| None, 0), 6);
        anchor.remember(0, || None);
        assert_eq!(anchor.restore(0, false, false, |_| Some(3), 0), 3);
        anchor.remember(3, || None);
        // Manual scroll replaces the preferred identity.
        assert_eq!(anchor.restore(1, false, false, |_| Some(3), 0), 1);
        anchor.remember(1, || Some((1, 99)));
        assert_eq!(anchor.restore(1, false, false, |_| None, 0), 0);
    }
    #[test]
    fn independent_viewports_and_scrollbar_bounds() {
        let mut state = Workspace::default();
        let list = state.viewport(3, Pane::Main);
        list.measure(100, 10);
        list.reveal(30);
        assert_eq!(list.offset, 21);
        let bar = Scrollbar::new(Pane::Main, Rect::new(10, 2, 1, 10), *list).unwrap();
        assert_eq!(bar.offset(99, 0), 90);
        assert_eq!(bar.offset(0, 0), 0);
        let detail = state.viewport(3, Pane::Detail);
        detail.measure(30, 5);
        detail.scroll(Motion::End);
        assert_eq!(detail.offset, 25);
        assert_eq!(state.viewport(3, Pane::Main).offset, 21);
    }
}
