//! Protocol-neutral text intent. Roles carry meaning, never template tokens.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatRole {
    System,
    Developer,
    User,
    Assistant,
}
impl ChatRole {
    pub fn parse(role: &str) -> Option<Self> {
        Some(match role {
            "system" => Self::System,
            "developer" => Self::Developer,
            "user" => Self::User,
            "assistant" => Self::Assistant,
            _ => return None,
        })
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Developer => "developer",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub text: String,
}

pub const MAX_CHAT_MESSAGES: usize = 128;

/// Raw text, single-user convenience or a structured conversation. No model
/// family or protocol-specific framing. Future content kinds require their
/// own capability and resource contracts, not coerced placeholder strings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextInput {
    Raw(String),
    User(String),
    Conversation(Vec<ChatMessage>),
}
impl TextInput {
    pub fn from_text(text: String, chat: bool) -> Self {
        if chat {
            Self::User(text)
        } else {
            Self::Raw(text)
        }
    }

    pub fn is_chat(&self) -> bool {
        !matches!(self, Self::Raw(_))
    }

    pub fn single_text(&self) -> Option<(&str, bool)> {
        match self {
            Self::Raw(text) => Some((text, false)),
            Self::User(text) => Some((text, true)),
            Self::Conversation(_) => None,
        }
    }

    /// Semantic content bytes, excluding the selected model's framing.
    pub fn content_bytes(&self) -> Option<usize> {
        match self {
            Self::Raw(text) | Self::User(text) => Some(text.len()),
            Self::Conversation(turns) => turns
                .iter()
                .try_fold(0usize, |n, turn| n.checked_add(turn.text.len())),
        }
    }

    /// Existing owned capacity, not length; queue admission includes the
    /// turn vector and all retained text backing.
    pub fn heap_bytes(&self) -> Option<usize> {
        match self {
            Self::Raw(text) | Self::User(text) => Some(text.capacity()),
            Self::Conversation(turns) => turns.iter().try_fold(
                turns
                    .capacity()
                    .checked_mul(std::mem::size_of::<ChatMessage>())?,
                |n, turn| n.checked_add(turn.text.capacity()),
            ),
        }
    }
}
