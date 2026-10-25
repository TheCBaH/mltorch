(* What a kernel is made of, in order: directives, labels, instructions and
   annotations. *)

module At = struct
  (* Where an item is: its position and the annotation in effect there. *)
  type t = { index : int; note : string option }

  let pp ppf { index; note } =
    match note with
    | None -> Fmt.pf ppf "item %d" index
    | Some note -> Fmt.pf ppf "item %d (%s)" index note
end

(* [Note (Some s)] annotates the items after it until the next [Note]; [Note
   None] clears the annotation. *)
type t =
  | Align of int
  | Bss
  | Export of { name : string; label : Rv32_kernel_label.t }
  | Insn of Rv32_kernel_insn.t
  | Label of Rv32_kernel_label.t
  | Note of string option
  | Text
  | Zero of int

let align boundary = Align boundary
let bss = Bss
let clear_note = Note None
let export ~name label = Export { name; label }
let insn i = Insn i
let label l = Label l
let note s = Note (Some s)
let text = Text
let zero length = Zero length
