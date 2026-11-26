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
    Owner(Owner),
    Shelf(Shelf),
    Split(Splitter),
    Scrollbar(Scrollbar),
    Maximize(Pane),
    Focus(Pane),
    Appearance,
    Theme(ThemeMode),
    Help,
    Palette,
    More,
    Command(Bound),
    AcceptModal,
    DismissModal,
    PromptMode,
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
    Split(Splitter),
    Scroll(Scrollbar, u16),
}
#[derive(Debug, Default)]
pub(super) struct Pointer(Option<Capture>);
pub(super) enum Intent {
    Click(Target),
    Resize(Splitter, u16, u16),
    Scroll(Pane, Motion),
    ScrollModal(i16, u16),
}
impl Pointer {
    pub fn cancel(&mut self) {
        self.0 = None;
    }
    pub fn invalidate_selection(&mut self) {
        if matches!(
            self.0,
            Some(Capture::Press(
                Target::Row(..) | Target::Owner(_) | Target::Command(_)
            ))
        ) {
            self.cancel();
        }
    }
    pub fn event(&mut self, event: MouseEvent, hits: &Hits) -> Option<Intent> {
        let pos = Position::new(event.column, event.row);
        match event.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                self.cancel();
                match hits.at(pos)?.clone() {
                    Target::Split(split) => self.0 = Some(Capture::Split(split)),
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
