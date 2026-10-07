(* Where each SSA buffer lives in Machine IR: one region and one view per
   buffer, sized by its format's element bytes with checked arithmetic. The
   view carries the buffer's source identity; distinct buffers get distinct
   regions here, which is a binding choice of this layout and never an alias
   proof the lowering relies on. *)

open Ssa_ir
open Machine_ir

module Entry = struct
  type t = {
    buffer : Ssa_buffer.t;
    region : Mir_id.Region.t;
    view : Mir_id.View.t;
    elem_bytes : int64;
    elements : int64;
  }
end

let elem_bytes = function
  | Ssa_format.Bool | Ssa_format.I8 _ -> 1L
  | Ssa_format.Bf16 | Ssa_format.F16 | Ssa_format.I16 _ -> 2L
  | Ssa_format.F32 | Ssa_format.I32 -> 4L
  | Ssa_format.F64 | Ssa_format.I64 -> 8L

let entries (buffers : Ssa_buffer.t list) =
  let rec go k acc = function
    | [] -> Ok (List.rev acc)
    | (b : Ssa_buffer.t) :: rest -> (
        let bytes = elem_bytes b.Ssa_buffer.format in
        match Ssa_buffer.elements b.Ssa_buffer.extents with
        | None -> Error b.Ssa_buffer.id
        | Some n -> (
            match Mir_layout.mul n bytes with
            | None -> Error b.Ssa_buffer.id
            | Some _ ->
                let e =
                  {
                    Entry.buffer = b;
                    region = Mir_id.Region.of_int k;
                    view = Mir_id.View.of_int k;
                    elem_bytes = bytes;
                    elements = n;
                  }
                in
                go (k + 1) (e :: acc) rest))
  in
  go 0 [] buffers

let size (e : Entry.t) = Int64.mul e.Entry.elements e.Entry.elem_bytes

let region (e : Entry.t) =
  {
    Mir_region.id = e.Entry.region;
    size = size e;
    align = 16L;
    init =
      (match e.Entry.buffer.Ssa_buffer.role with
      | Ssa_buffer.Input | Ssa_buffer.Output -> Mir_region.Bound
      | Ssa_buffer.Scratch -> Mir_region.Uninitialized);
  }

let view (e : Entry.t) =
  let role, perm =
    match e.Entry.buffer.Ssa_buffer.role with
    | Ssa_buffer.Input -> (Mir_view.Input, Mir_view.Read)
    | Ssa_buffer.Output -> (Mir_view.Output, Mir_view.Read_write)
    | Ssa_buffer.Scratch -> (Mir_view.Scratch, Mir_view.Read_write)
  in
  {
    Mir_view.id = e.Entry.view;
    region = e.Entry.region;
    offset = 0L;
    size = size e;
    perm;
    role;
    source = Some (Ssa_buffer.source e.Entry.buffer);
  }

let find entries id =
  List.find_opt
    (fun (e : Entry.t) -> Ssa_id.Buffer.equal e.Entry.buffer.Ssa_buffer.id id)
    entries
