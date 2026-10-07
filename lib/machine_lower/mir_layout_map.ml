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

(* A per-channel quantized buffer's parameter table: a constant region of the
   scales (binary64 bits) then the zero points (i64), eight bytes per channel
   each, read through a view of its own. *)
module Params = struct
  let base = 100_000

  let view (e : Entry.t) =
    match e.Entry.buffer.Ssa_buffer.format with
    | Ssa_format.I8 (Ssa_format.Per_channel _)
    | Ssa_format.I16 (Ssa_format.Per_channel _) ->
        Some (Mir_id.View.of_int (base + Mir_id.View.to_int e.Entry.view))
    | _ -> None

  let channels (e : Entry.t) =
    Option.value ~default:0
      (Ssa_format.channels e.Entry.buffer.Ssa_buffer.format)

  (* the byte offset of channel parameters: scales first, zero points after *)
  let zero_point_offset e = Int64.mul 8L (Int64.of_int (channels e))

  let bytes (e : Entry.t) =
    match Ssa_format.quant e.Entry.buffer.Ssa_buffer.format with
    | Some (Ssa_format.Per_channel { scale; zero_point }) ->
        let b = Bytes.create (16 * Array.length scale) in
        Array.iteri
          (fun i s -> Bytes.set_int64_le b (8 * i) (Int64.bits_of_float s))
          scale;
        Array.iteri
          (fun i z ->
            Bytes.set_int64_le b (8 * (Array.length scale + i)) (Int64.of_int z))
          zero_point;
        Some (Bytes.to_string b)
    | Some (Ssa_format.Per_tensor _) | None -> None

  let objects (e : Entry.t) =
    match (view e, bytes e) with
    | Some v, Some s ->
        let region =
          Mir_id.Region.of_int (base + Mir_id.Region.to_int e.Entry.region)
        in
        let size = Int64.of_int (String.length s) in
        [
          ( {
              Mir_region.id = region;
              size;
              align = 8L;
              init = Mir_region.Constant s;
            },
            {
              Mir_view.id = v;
              region;
              offset = 0L;
              size;
              perm = Mir_view.Read;
              role = Mir_view.Constant;
              source = None;
            } );
        ]
    | _ -> []
end

(* Invocation storage beside the buffers: one region per local allocation site,
   and the scan meter's two words. A site's object is reused by every execution
   of the site, which is sound only because the lowering refuses a local that
   reaches a use through anything but its own allocation, so no earlier
   instance is reachable once the site runs again; each run begins with the
   object's bytes undefined. *)
module Scratch = struct
  let local_base = 200_000
  let meter_id = 300_000

  (* the most bytes every local site of one program may hold together *)
  let local_limit = 0x8000_0000L

  let object_ id ~size ~role =
    let region = Mir_id.Region.of_int id in
    ( {
        Mir_region.id = region;
        size;
        align = 16L;
        init = Mir_region.Uninitialized;
      },
      {
        Mir_view.id = Mir_id.View.of_int id;
        region;
        offset = 0L;
        size;
        perm = Mir_view.Read_write;
        role;
        source = None;
      } )

  (* the [k]th local site's object of [bytes] *)
  let local k ~bytes =
    object_ (local_base + k) ~size:bytes ~role:Mir_view.Scratch

  (* updates remaining at byte 0, live state at byte 8, both i64 *)
  let meter = object_ meter_id ~size:16L ~role:Mir_view.Runtime
  let meter_remaining = 0L
  let meter_live = 8L
end
