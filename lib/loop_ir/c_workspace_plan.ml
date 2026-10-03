open Graph_ir
module S = Storage_script
module U = Core.Storage_units

type location = Inputs of int64 | Weights of int64 | Workspace of int64

type error =
  [ `Config_unsupported of S.Config.t
  | `Edge_unplaced of Tensor_id.t
  | `Slot_out_of_pool of Tensor_id.t
  | `Storage_units of U.error
  | `Synthetic_not_f32 of Tensor_id.t
  | `Workspace_overflow ]

let pp_error ppf : [< error ] -> unit = function
  | `Config_unsupported c ->
      Format.fprintf ppf
        "storage config %a: the whole-model backends admit only separate \
         layout with borrowed constants and inputs"
        S.Config.pp c
  | `Edge_unplaced id ->
      Format.fprintf ppf "t%d: no arena slot and no payload entry"
        (Tensor_id.to_int id)
  | `Slot_out_of_pool id ->
      Format.fprintf ppf "t%d: arena slot lies outside its pool"
        (Tensor_id.to_int id)
  | `Storage_units e -> U.pp_error ppf e
  | `Synthetic_not_f32 id ->
      Format.fprintf ppf "t%d: a synthetic default that is not F32"
        (Tensor_id.to_int id)
  | `Workspace_overflow -> Format.fprintf ppf "workspace size overflows"

let ceiling = Int64.shift_left 1L 60
let base_alignment = 64L
let align_up n a = Int64.mul (Int64.div (Int64.add n (Int64.pred a)) a) a

let add a b =
  let r = Int64.add a b in
  if Int64.compare r ceiling > 0 || Int64.compare r 0L < 0 then
    Err.fail `Workspace_overflow
  else Err.return r

module Scratch = struct
  type fill = Zero | Value of float
  type carve = { position : int; offset : int64; bytes : int64; fill : fill }
  type t = { local_bytes : int64; carves : carve list; bytes : int64 }
end

let sig_bytes (sg : Tensor_sig.t) =
  let cell = Int64.of_int (Payload.packed_cell_bytes sg.Tensor_sig.fmt) in
  match
    Vec6.numel_bounded ~limit:(Int64.shift_left 1L 40) sg.Tensor_sig.shape
  with
  | Ok n -> Err.return (Int64.mul n cell)
  | Error _ -> Err.fail `Workspace_overflow

let scratch (inv : Loop_bundle.invocation) ~local_doubles =
  let open Err.Syntax in
  let local_bytes = Int64.mul local_doubles 8L in
  let* start = add (align_up local_bytes base_alignment) 0L in
  let synthetic id =
    List.find_opt
      (fun (s : Loop_bundle.synthetic) -> Tensor_id.equal s.Loop_bundle.id id)
      inv.Loop_bundle.synthetics
  in
  let+ carves, total =
    Err.List.fold_left
      (fun (acc, off) (position, ((buf : Loop_buffer.t), edge)) ->
        match (synthetic edge, buf.Loop_buffer.role) with
        | Some s, _ ->
            let* () =
              match buf.Loop_buffer.sg.Tensor_sig.fmt with
              | Payload.Fmt Payload.F32 -> Err.return ()
              | Payload.Fmt _ -> Err.fail (`Synthetic_not_f32 edge)
            in
            let* bytes = sig_bytes buf.Loop_buffer.sg in
            let+ next = add off (align_up bytes base_alignment) in
            ( {
                Scratch.position;
                offset = off;
                bytes;
                fill = Scratch.Value s.Loop_bundle.value;
              }
              :: acc,
              next )
        | None, Loop_buffer.Scratch ->
            let* bytes = sig_bytes buf.Loop_buffer.sg in
            let+ next = add off (align_up bytes base_alignment) in
            ( { Scratch.position; offset = off; bytes; fill = Scratch.Zero }
              :: acc,
              next )
        | None, (Loop_buffer.Input | Loop_buffer.Output) -> Err.return (acc, off))
      ([], start)
      (List.mapi
         (fun i x -> (i, x))
         (List.combine inv.Loop_bundle.program.Loop_program.buffers
            inv.Loop_bundle.edges))
  in
  { Scratch.local_bytes; carves = List.rev carves; bytes = total }

module Pool = struct
  type t = {
    arena : S.Arena_id.t;
    kind : Alloc_script.Kind.t;
    offset : int64;
    bytes : int64;
  }
end

type t = {
  pools : Pool.t list;
  locations : (Tensor_id.t, location) Hashtbl.t;
  scratch_offset : int64;
  total : int64;
  alignment : int64;
  arena_bytes : int64;
}

let create (b : Loop_bundle.t) ~weights ~inputs ~scratch_bytes =
  let open Err.Syntax in
  let cfg = b.Loop_bundle.config in
  let* () =
    if
      cfg.S.Config.layout = S.Layout.Separate
      && cfg.S.Config.constants = S.Ownership.Borrowed
      && cfg.S.Config.inputs = S.Ownership.Borrowed
    then Err.return ()
    else Err.fail (`Config_unsupported cfg)
  in
  let locations = Hashtbl.create 256 in
  List.iter
    (fun (e : C_payload_layout.Entry.t) ->
      Hashtbl.replace locations e.C_payload_layout.Entry.id
        (Weights e.C_payload_layout.Entry.offset))
    weights.C_payload_layout.entries;
  List.iter
    (fun (e : C_payload_layout.Entry.t) ->
      Hashtbl.replace locations e.C_payload_layout.Entry.id
        (Inputs e.C_payload_layout.Entry.offset))
    inputs.C_payload_layout.entries;
  let bytes_of x = U.Byte_size.to_int64 x in
  let* pools, cursor, alignment =
    Err.List.fold_left
      (fun (pools, off, align) (arena, plan) ->
        Err.List.fold_left
          (fun (pools, off, align) (p : Arena_plan.Pool.t) ->
            let a =
              Int64.max base_alignment
                (U.Byte_alignment.to_int64 p.Arena_plan.Pool.alignment)
            in
            let start = align_up off a in
            let size = bytes_of p.Arena_plan.Pool.bytes in
            let+ next = add start size in
            ( {
                Pool.arena;
                kind = p.Arena_plan.Pool.kind;
                offset = start;
                bytes = size;
              }
              :: pools,
              next,
              Int64.max align a ))
          (pools, off, align) (Arena_plan.pools plan))
      ([], 0L, base_alignment)
      (Storage_plan.arenas b.Loop_bundle.plan)
  in
  let pools = List.rev pools in
  let arena_bytes = cursor in
  (* Every arena slot, checked against the pool it claims to sit in. *)
  let* () =
    Err.List.iter
      (fun (arena, plan) ->
        Err.List.iter
          (fun (s : Arena_plan.Slot.t) ->
            match
              List.find_opt
                (fun (p : Pool.t) ->
                  p.Pool.arena = arena && p.Pool.kind = s.Arena_plan.Slot.kind)
                pools
            with
            | None -> Err.fail (`Slot_out_of_pool s.Arena_plan.Slot.id)
            | Some p ->
                let off = U.Byte_offset.to_int64 s.Arena_plan.Slot.offset in
                let size = bytes_of s.Arena_plan.Slot.bytes in
                let* stop = add off size in
                let elem =
                  U.Element_bytes.to_int64
                    (Alloc_script.Kind.element_bytes s.Arena_plan.Slot.kind)
                in
                if
                  Int64.compare off 0L < 0
                  || Int64.compare stop p.Pool.bytes > 0
                  || not (Int64.equal (Int64.rem off elem) 0L)
                then Err.fail (`Slot_out_of_pool s.Arena_plan.Slot.id)
                else (
                  Hashtbl.replace locations s.Arena_plan.Slot.id
                    (Workspace (Int64.add p.Pool.offset off));
                  Err.return ()))
          (Arena_plan.slots plan))
      (Storage_plan.arenas b.Loop_bundle.plan)
  in
  let scratch_offset = align_up cursor base_alignment in
  let+ total = add scratch_offset scratch_bytes in
  { pools; locations; scratch_offset; total; alignment; arena_bytes }

let locate t id =
  match Hashtbl.find_opt t.locations id with
  | Some l -> Err.return l
  | None -> Err.fail (`Edge_unplaced id)

let scratch_offset t = t.scratch_offset
let bytes t = t.total
let alignment t = t.alignment
let graph_arena_bytes t = t.arena_bytes

let pools t =
  List.map
    (fun (p : Pool.t) ->
      (p.Pool.arena, p.Pool.kind, p.Pool.offset, p.Pool.bytes))
    t.pools
