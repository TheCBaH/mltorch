#!/usr/bin/env python3
"""Prints "<dir>\t<library>\t<mode>" for every (inline_tests) library's
dune stanza in the project, excluding vendored/ and modules/."""

import subprocess
import sys


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c == ";":
            # dune line comment: skip to end of line.
            j = text.find("\n", i)
            i = n if j == -1 else j + 1
            continue
        if c.isspace():
            i += 1
            continue
        if c in "()":
            tokens.append(c)
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                j += 1
            tokens.append(text[i : j + 1])
            i = j + 1
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in "()":
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def parse(tokens: list[str], pos: int):
    """Parse one s-expr starting at tokens[pos]; return (value, next_pos)."""
    tok = tokens[pos]
    if tok != "(":
        return tok, pos + 1
    pos += 1
    items = []
    while tokens[pos] != ")":
        item, pos = parse(tokens, pos)
        items.append(item)
    return items, pos + 1


def parse_all(text: str):
    tokens = tokenize(text)
    forms = []
    pos = 0
    while pos < len(tokens):
        form, pos = parse(tokens, pos)
        forms.append(form)
    return forms


def unquote(atom: str) -> str:
    return atom[1:-1] if len(atom) >= 2 and atom[0] == '"' and atom[-1] == '"' else atom


def find_field(form: list, field_name: str):
    """Find `(field_name ...)` among form's direct children; return its
    argument list (everything after the head), or None."""
    for item in form:
        if isinstance(item, list) and item and item[0] == field_name:
            return item[1:]
    return None


def modes_of(inline_tests_form: list) -> list[str]:
    modes_field = find_field(inline_tests_form, "modes")
    if modes_field is None:
        return ["best"]
    return [unquote(m) for m in modes_field]


def main() -> int:
    out = subprocess.run(
        ["git", "ls-files", "--", "*dune"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for path in out.splitlines():
        if path.startswith("vendored/") or path.startswith("modules/"):
            continue
        if path.split("/")[-1] != "dune":
            continue
        directory = path.rsplit("/", 1)[0] if "/" in path else "."
        with open(path, encoding="utf-8") as f:
            text = f.read()
        try:
            forms = parse_all(text)
        except IndexError:
            print(f"warning: failed to parse {path}", file=sys.stderr)
            continue
        for form in forms:
            if not (isinstance(form, list) and form and form[0] == "library"):
                continue
            name_field = find_field(form, "name")
            inline_tests_field = None
            for item in form:
                if isinstance(item, list) and item and item[0] == "inline_tests":
                    inline_tests_field = item
                    break
            if name_field is None or inline_tests_field is None:
                continue
            lib = unquote(name_field[0])
            for mode in modes_of(inline_tests_field):
                print(f"{directory}\t{lib}\t{mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
