//! Narrow glibc ELF64 bootstrap, without calling interposed dlsym/dlvsym.
use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;
#[repr(C)]
struct Phdr {
    kind: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    paddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}
#[repr(C)]
struct Info {
    base: usize,
    name: *const c_char,
    phdr: *const Phdr,
    phnum: u16,
}
#[repr(C)]
struct Dynamic {
    tag: i64,
    value: usize,
}
#[repr(C)]
struct Symbol {
    name: u32,
    info: u8,
    other: u8,
    section: u16,
    value: usize,
    size: usize,
}
unsafe extern "C" {
    fn dl_iterate_phdr(
        callback: unsafe extern "C" fn(*mut Info, usize, *mut c_void) -> c_int,
        data: *mut c_void,
    ) -> c_int;
}
struct Query<'a> {
    name: &'a [u8],
    pointer: *mut c_void,
}
unsafe fn contains(info: &Info, start: usize, bytes: usize) -> bool {
    let Some(end) = start.checked_add(bytes) else {
        return false;
    };
    for index in 0..usize::from(info.phnum) {
        let phdr = unsafe { &*info.phdr.add(index) };
        if phdr.kind != 1 {
            continue;
        }
        let Some(low) = info.base.checked_add(phdr.vaddr as usize) else {
            continue;
        };
        let Some(high) = low.checked_add(phdr.memsz as usize) else {
            continue;
        };
        if start >= low && end <= high {
            return true;
        }
    }
    false
}
unsafe extern "C" fn visit(info: *mut Info, size: usize, opaque: *mut c_void) -> c_int {
    if info.is_null() || size < std::mem::size_of::<Info>() {
        return 0;
    }
    let info = unsafe { &*info };
    if info.name.is_null() || info.phdr.is_null() || info.phnum == 0 || info.phnum > 128 {
        return 0;
    }
    let name = unsafe { CStr::from_ptr(info.name) }.to_bytes();
    if name.rsplit(|byte| *byte == b'/').next() != Some(b"libc.so.6") {
        return 0;
    }
    let mut dynamic = ptr::null::<Dynamic>();
    let mut count = 0;
    for index in 0..usize::from(info.phnum) {
        let phdr = unsafe { &*info.phdr.add(index) };
        if phdr.kind == 2 {
            dynamic = (info.base + phdr.vaddr as usize) as *const Dynamic;
            count = (phdr.memsz as usize / std::mem::size_of::<Dynamic>()).min(4096);
        }
    }
    if dynamic.is_null()
        || !unsafe {
            contains(
                info,
                dynamic as usize,
                count * std::mem::size_of::<Dynamic>(),
            )
        }
    {
        return 1;
    }
    let (mut hash, mut symbols, mut strings, mut string_bytes, mut symbol_bytes) = (0, 0, 0, 0, 0);
    for index in 0..count {
        let entry = unsafe { &*dynamic.add(index) };
        if entry.tag == 0 {
            break;
        }
        match entry.tag {
            4 => hash = entry.value,
            5 => strings = entry.value,
            6 => symbols = entry.value,
            10 => string_bytes = entry.value,
            11 => symbol_bytes = entry.value,
            _ => {}
        }
    }
    // glibc relocates these dynamic pointers. Require DT_HASH and exact ELF64
    // entries; unsupported layouts fail closed rather than guessing a resolver.
    if symbol_bytes != std::mem::size_of::<Symbol>()
        || !unsafe { contains(info, hash, 8) }
        || !unsafe { contains(info, strings, string_bytes) }
    {
        return 1;
    }
    let symbol_count = unsafe { *((hash as *const u32).add(1)) } as usize;
    if symbol_count > 2_000_000 || !unsafe { contains(info, symbols, symbol_count * symbol_bytes) }
    {
        return 1;
    }
    let query = unsafe { &mut *(opaque as *mut Query<'_>) };
    for index in 0..symbol_count {
        let symbol = unsafe { &*(symbols as *const Symbol).add(index) };
        let offset = symbol.name as usize;
        if symbol.section == 0 || symbol.info & 15 != 2 || offset >= string_bytes {
            continue;
        }
        let Some(end) = offset.checked_add(query.name.len() + 1) else {
            continue;
        };
        if end > string_bytes {
            continue;
        }
        let candidate = unsafe {
            std::slice::from_raw_parts((strings + offset) as *const u8, query.name.len() + 1)
        };
        if candidate[..query.name.len()] == *query.name && candidate[query.name.len()] == 0 {
            let Some(pointer) = info.base.checked_add(symbol.value) else {
                return 1;
            };
            if unsafe { contains(info, pointer, 1) } {
                query.pointer = pointer as *mut c_void;
            }
            return 1;
        }
    }
    1
}
pub(super) fn glibc_symbol(name: &[u8]) -> *mut c_void {
    let mut query = Query {
        name,
        pointer: ptr::null_mut(),
    };
    unsafe { dl_iterate_phdr(visit, (&mut query as *mut Query<'_>).cast()) };
    query.pointer
}
