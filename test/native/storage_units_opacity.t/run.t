Unit separation: what the type checker must reject.

Byte and element quantities, sizes and offsets, and sizes and alignments are
distinct abstract types, so mixing them is a compile error rather than a wrong
number. Every negative case is paired with a control that must still compile --
without the controls a broken harness would reject everything and look like a
pass.

  $ R=../../..
  $ check() { sh $R/test/native/storage_units_opacity.sh $R/test/native/core_probe.exe "$1" "$2"; }

Bytes are not elements.

  $ check "add an element count to a byte size" "ignore (Byte_size.add bs ec)"
  add an element count to a byte size: rejected
  $ check "add two byte sizes" "ignore (Byte_size.add bs bs)"
  add two byte sizes: COMPILES
  $ check "advance a byte offset by elements" "ignore (Byte_offset.advance bo ec)"
  advance a byte offset by elements: rejected
  $ check "advance an element offset by elements" "ignore (Element_offset.advance eo ec)"
  advance an element offset by elements: COMPILES
  $ check "compare a byte offset with an element offset" "ignore (Byte_offset.equal bo eo)"
  compare a byte offset with an element offset: rejected
  $ check "convert an element offset to bytes" "ignore (Element_offset.to_bytes eo ew : (Byte_offset.t, _) Err.t)"
  convert an element offset to bytes: COMPILES

A size is not an offset, and two offsets do not add.

  $ check "pass a size for an offset" "ignore (Byte_offset.is_aligned bs ba)"
  pass a size for an offset: rejected
  $ check "advance an offset by an offset" "ignore (Byte_offset.advance bo bo)"
  advance an offset by an offset: rejected
  $ check "advance an offset by a size" "ignore (Byte_offset.advance bo bs)"
  advance an offset by a size: COMPILES
  $ check "size an element count by an offset" "ignore (Element_count.to_bytes eo ew)"
  size an element count by an offset: rejected

A size is not an alignment, except through the named conversion.

  $ check "align by a size" "ignore (Byte_offset.align_up bo bs)"
  align by a size: rejected
  $ check "align by an alignment" "ignore (Byte_offset.align_up bo ba)"
  align by an alignment: COMPILES
  $ check "compare an alignment with a size" "ignore (Byte_size.equal ba bs)"
  compare an alignment with a size: rejected
  $ check "compare through to_size" "ignore (Byte_size.equal (Byte_alignment.to_size ba) bs)"
  compare through to_size: COMPILES

A raw [int64] enters only through a checked constructor; the exit is named.

  $ check "pass an int64 for a size" "ignore (Byte_size.add bs n)"
  pass an int64 for a size: rejected
  $ check "coerce an int64 into a size" "ignore ((n :> Byte_size.t))"
  coerce an int64 into a size: rejected
  $ check "coerce a size out to int64" "ignore ((bs :> int64))"
  coerce a size out to int64: rejected
  $ check "enter through of_int64" "ignore (Byte_size.of_int64 n)"
  enter through of_int64: COMPILES
  $ check "exit through to_int64" "ignore (Int64.add (Byte_size.to_int64 bs) n)"
  exit through to_int64: COMPILES
