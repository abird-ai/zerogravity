//! Client-local semantic colors. No terminal queries in the input loop.
use ratatui::style::{Color, Modifier, Style};
use zerogravity::configuration::ThemeMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Depth {
    TrueColor,
    Ansi256,
    Ansi16,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Environment {
    depth: Depth,
    auto_light: bool,
}
impl Environment {
    pub fn detect() -> Self {
        Self::from_hints(
            std::env::var("COLORTERM").ok().as_deref(),
            std::env::var("TERM").ok().as_deref(),
            std::env::var("COLORFGBG").ok().as_deref(),
        )
    }
    pub(super) fn from_hints(
        color: Option<&str>,
        term: Option<&str>,
        ground: Option<&str>,
    ) -> Self {
        let depth = match color {
            Some("truecolor" | "24bit") => Depth::TrueColor,
            _ if term.is_some_and(|t| t.contains("256color")) => Depth::Ansi256,
            _ => Depth::Ansi16,
        };
        // No speculative OSC exchange: unsupported/remote terminals may
        // deliver delayed responses as keys. Until a bounded local probe
        // is qualified, an unknown background deterministically uses dark.
        let auto_light = ground
            .and_then(|value| value.rsplit(';').next())
            .and_then(|value| value.parse::<u8>().ok())
            .is_some_and(|bg| bg == 7 || (9..=15).contains(&bg));
        Self { depth, auto_light }
    }
    pub fn resolve(self, mode: ThemeMode, color: bool) -> Theme {
        let light = match mode {
            ThemeMode::Auto => self.auto_light,
            ThemeMode::Dark => false,
            ThemeMode::Light => true,
        };
        let rgb = if light { LIGHT } else { DARK };
        let colors = if !color {
            [Color::Reset; ROLE_COUNT]
        } else {
            match self.depth {
                Depth::TrueColor => rgb.map(|[r, g, b]| Color::Rgb(r, g, b)),
                Depth::Ansi256 => rgb.map(indexed),
                Depth::Ansi16 => {
                    if light {
                        LIGHT_16
                    } else {
                        DARK_16
                    }
                }
            }
        };
        Theme {
            colors,
            mode,
            light,
            color,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Theme {
    colors: [Color; ROLE_COUNT],
    mode: ThemeMode,
    light: bool,
    color: bool,
}
#[derive(Clone, Copy)]
enum Role {
    Canvas,
    Surface,
    Raised,
    Selection,
    Text,
    Muted,
    Rule,
    Accent,
    Success,
    Warning,
    Danger,
    Info,
}
const ROLE_COUNT: usize = 12;
const DARK: [[u8; 3]; ROLE_COUNT] = [
    [0x11, 0x13, 0x18], // canvas
    [0x17, 0x1a, 0x20], // surface
    [0x1e, 0x23, 0x2b], // raised
    [0x23, 0x37, 0x35], // selection
    [0xe8, 0xeb, 0xf0], // text
    [0x92, 0x9e, 0xad], // muted
    [0x34, 0x3d, 0x48], // rule
    [0x80, 0xd1, 0xc5], // accent
    [0x83, 0xcf, 0xa8], // success
    [0xe5, 0xbc, 0x79], // warning
    [0xf1, 0x97, 0x9c], // danger
    [0x9d, 0xb8, 0xe6], // info
];
const LIGHT: [[u8; 3]; ROLE_COUNT] = [
    [0xf5, 0xf6, 0xf8],
    [0xff, 0xff, 0xff],
    [0xee, 0xf1, 0xf5],
    [0xdc, 0xef, 0xea],
    [0x26, 0x31, 0x3c],
    [0x59, 0x68, 0x79],
    [0xd7, 0xde, 0xe6],
    [0x11, 0x6b, 0x63],
    [0x20, 0x6c, 0x4b],
    [0x8a, 0x56, 0x0e],
    [0xa3, 0x37, 0x43],
    [0x31, 0x5e, 0x99],
];
// Limited palettes use predictable contrasting surfaces, not RGB assumptions.
const DARK_16: [Color; ROLE_COUNT] = [
    Color::Black,
    Color::Black,
    Color::Black,
    Color::DarkGray,
    Color::White,
    Color::Gray,
    Color::DarkGray,
    Color::LightCyan,
    Color::LightGreen,
    Color::LightYellow,
    Color::LightRed,
    Color::LightBlue,
];
const LIGHT_16: [Color; ROLE_COUNT] = [
    Color::Gray,
    Color::Gray,
    Color::White,
    Color::White,
    Color::Black,
    Color::Black,
    Color::DarkGray,
    Color::Blue,
    Color::Green,
    Color::Red,
    Color::Red,
    Color::Blue,
];

// Nearest fixed xterm cube/gray; never guess the user-defined first 16.
fn indexed(rgb: [u8; 3]) -> Color {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let distance = |a: [u8; 3], b: [u8; 3]| -> u32 {
        a.into_iter()
            .zip(b)
            .map(|(a, b)| (i32::from(a) - i32::from(b)).pow(2) as u32)
            .sum()
    };
    let nearest = |v: u8| {
        LEVELS
            .iter()
            .enumerate()
            .min_by_key(|(_, level)| v.abs_diff(**level))
            .unwrap()
            .0
    };
    let [r, g, b] = rgb.map(nearest);
    let cube = 16 + 36 * r + 6 * g + b;
    let cube_rgb = [LEVELS[r], LEVELS[g], LEVELS[b]];
    let gray = (0..24)
        .min_by_key(|i| distance(rgb, [8 + 10 * i; 3]))
        .unwrap();
    let index = if distance(rgb, [8 + 10 * gray; 3]) < distance(rgb, cube_rgb) {
        232 + usize::from(gray)
    } else {
        cube
    };
    Color::Indexed(index as u8)
}

impl Theme {
    fn fg(self, role: Role) -> Style {
        if self.color {
            Style::new().fg(self.colors[role as usize])
        } else {
            Style::new()
        }
    }
    fn on(self, background: Role) -> Style {
        let style = self.fg(Role::Text);
        if self.color {
            style.bg(self.colors[background as usize])
        } else {
            style
        }
    }
    pub fn canvas(self) -> Style {
        self.on(Role::Canvas)
    }
    pub fn surface(self) -> Style {
        self.on(Role::Surface)
    }
    pub fn raised(self) -> Style {
        self.on(Role::Raised)
    }
    pub fn heading(self) -> Style {
        self.fg(Role::Text).add_modifier(Modifier::BOLD)
    }
    pub fn accent(self) -> Style {
        self.fg(Role::Accent).add_modifier(Modifier::BOLD)
    }
    pub fn muted(self) -> Style {
        self.fg(Role::Muted)
    }
    pub fn border(self, focused: bool) -> Style {
        if focused {
            self.fg(Role::Accent)
        } else {
            self.fg(Role::Rule)
        }
    }
    pub fn selected(self) -> Style {
        if self.color {
            self.on(Role::Selection).add_modifier(Modifier::BOLD)
        } else {
            Style::new().add_modifier(Modifier::BOLD | Modifier::REVERSED)
        }
    }
    pub fn success(self) -> Style {
        self.fg(Role::Success)
    }
    pub fn warning(self) -> Style {
        self.fg(Role::Warning)
    }
    pub fn danger(self) -> Style {
        self.fg(Role::Danger).add_modifier(Modifier::BOLD)
    }
    pub fn info(self) -> Style {
        self.fg(Role::Info)
    }
    pub fn description(self) -> &'static str {
        match (self.mode, self.light) {
            (ThemeMode::Auto, true) => "Auto · Light",
            (ThemeMode::Auto, false) => "Auto · Dark",
            (ThemeMode::Light, _) => "Light",
            (ThemeMode::Dark, _) => "Dark",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn appearance_fallback_and_color_policy() {
        let env = Environment::from_hints(Some("truecolor"), None, Some("0;15"));
        let auto = env.resolve(ThemeMode::Auto, true);
        assert_eq!(auto.description(), "Auto · Light");
        assert_eq!(auto.canvas().bg, Some(Color::Rgb(245, 246, 248)));
        assert_eq!(
            env.resolve(ThemeMode::Dark, true).canvas().bg,
            Some(Color::Rgb(17, 19, 24))
        );
        let plain = env.resolve(ThemeMode::Light, false);
        assert_eq!(plain.canvas(), Style::new());
        assert_eq!(plain.selected().bg, None);
        assert!(plain.selected().add_modifier.contains(Modifier::REVERSED));
        let unknown = Environment::from_hints(None, Some("xterm-256color"), None);
        assert_eq!(
            unknown.resolve(ThemeMode::Auto, true).description(),
            "Auto · Dark"
        );
        assert!(matches!(
            unknown.resolve(ThemeMode::Auto, true).canvas().bg,
            Some(Color::Indexed(16..=255))
        ));
        let limited = Environment::from_hints(None, None, Some("15;0"));
        assert_eq!(
            limited.resolve(ThemeMode::Auto, true).canvas().bg,
            Some(Color::Black)
        );
    }
}
