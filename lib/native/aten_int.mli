(* Integers as ATen wrote them: signed, unvalidated, and meaning different
   things. [-1] is a legal dim number (the last), a legal size (infer it) and a
   legal index (the last), yet none of the three is an extent, a position or an
   axis until [Aten_shape] has resolved it against a rank or an extent — the
   one place they are normalised. The types keep a dim as written from being
   read as a size, and neither from being read as a resolved extent.

   Error payloads report these as written, so a caller who wrote -9 sees -9.
   Entry is [of_int] (the importer's decode of an ATen argument); there is no
   check, because nothing about the value is known yet. Exit is the free
   coercion, for printing and for ATen's own bindings. *)

module Dim : Core.Tagged_int.S
(** A dim number: which axis of a tensor of some rank. *)

module Index : Core.Tagged_int.S
(** A position along one axis: a slice bound or a select index. *)

module Size : Core.Tagged_int.S
(** An entry of a size-like list: a view/expand size, a repeat count, a pad
    amount, a tensor's own dims. *)

module Step : Core.Tagged_int.S
(** A slice step. *)
