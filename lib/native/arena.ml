(* See arena.mli. *)

open Bigarray
open Core.Storage_units

(* Conversions the plan has already ruled out: a slot's offset is aligned to at
   least its cell width, and a pool's total bytes are far below [int64]. *)
let invariant r = Err.or_raise ~pp_error r

module Poison = struct
  type t = A | B
end

module Admission = struct
  type t = Best_effort | Required of Byte_size.t
end

module Over_budget = struct
  type t = { footprint : Byte_size.t; budget : Byte_size.t }
end

module Alloc_failed = struct
  type t = { kind : Alloc_script.Kind.t; bytes : Byte_size.t }
end

module Physical_alignment = struct
  type t = Logical_only

  let pp ppf Logical_only = Format.pp_print_string ppf "logical_only"
  let backend = Logical_only
end

module Physical_requirement = struct
  type t = Logical_accepted | Physical_required
end

module Physical_unsupported = struct
  type t = { required : Byte_alignment.t; provided : Physical_alignment.t }
end

type error =
  [ `Arena_alloc_failed of Alloc_failed.t
  | `Arena_busy
  | `Arena_script_mismatch of Alloc_script.Difference.t
  | `Arena_slot of Tensor_id.t
  | `Over_budget of Over_budget.t
  | `Physical_alignment_unsupported of Physical_unsupported.t
  | `Storage_script_mismatch of Alloc_script.Position.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Arena_alloc_failed { Alloc_failed.kind; bytes } ->
      Format.fprintf ppf "arena: cannot allocate the %a pool (%a bytes)"
        Alloc_script.Kind.pp kind Byte_size.pp bytes
  | `Arena_busy -> Format.pp_print_string ppf "arena: already in use by a run"
  | `Arena_script_mismatch { Alloc_script.Difference.position; left; right } ->
      let pp_event ppf = function
        | None -> Format.pp_print_string ppf "end of script"
        | Some e -> Alloc_script.Event.pp ppf e
      in
      Format.fprintf ppf
        "arena: the plan was built for a different run: at event %a the plan \
         has %a, this run has %a"
        Alloc_script.Position.pp position pp_event left pp_event right
  | `Arena_slot id ->
      Format.fprintf ppf "arena: the slot of t%d does not fit its format"
        (Tensor_id.to_int id)
  | `Over_budget { Over_budget.footprint; budget } ->
      Format.fprintf ppf "arena: the run needs %a bytes but the budget is %a"
        Byte_size.pp footprint Byte_size.pp budget
  | `Physical_alignment_unsupported { Physical_unsupported.required; provided }
    ->
      Format.fprintf ppf
        "arena: the pools need a base aligned to %a bytes, the backend gives %a"
        Byte_alignment.pp required Physical_alignment.pp provided
  | `Storage_script_mismatch position ->
      Format.fprintf ppf
        "arena: the storage plan was built for a different run: its script \
         differs at event %a"
        Alloc_script.Position.pp position

module Copies = struct
  type t = { count : int64; bytes : Byte_size.t }
end

type pool =
  | Float32 of (float, float32_elt, c_layout) Array1.t
  | Float64 of (float, float64_elt, c_layout) Array1.t
  | Int16_unsigned of (int, int16_unsigned_elt, c_layout) Array1.t
  | Int32 of (int32, int32_elt, c_layout) Array1.t
  | Int64 of (int64, int64_elt, c_layout) Array1.t
  | Int8_unsigned of (int, int8_unsigned_elt, c_layout) Array1.t

type t = {
  plan : Arena_plan.t;
  pools : (Alloc_script.Kind.t * pool) list;
  poison : Poison.t option;
  mutable busy : bool;
  mutable copies : Copies.t;
}

let plan t = t.plan
let copies t = t.copies
let physical_alignment _ = Physical_alignment.backend

(* A tally: no run copies 2^63 bytes. *)
let record_copy t ~bytes =
  t.copies <-
    {
      Copies.count = Int64.succ t.copies.Copies.count;
      bytes = invariant (Byte_size.add t.copies.Copies.bytes bytes);
    }

(* Poison values: finite, representable in every kind, different between [A] and
   [B]. Bool cells are the canonical true / false, since any nonzero byte reads
   as true and two poisons must differ there too. *)
let float_poison = function Poison.A -> 1.25e30 | Poison.B -> -3.5e29
let int_poison = function Poison.A -> 0x5A5A | Poison.B -> 0x3C3C

let fill pool poison =
  match pool with
  | Float32 a -> Array1.fill a (float_poison poison)
  | Float64 a -> Array1.fill a (float_poison poison)
  | Int16_unsigned a -> Array1.fill a (int_poison poison)
  | Int32 a -> Array1.fill a (Int32.of_int (int_poison poison))
  | Int64 a -> Array1.fill a (Int64.of_int (int_poison poison))
  | Int8_unsigned a ->
      Array1.fill a (match poison with Poison.A -> 1 | Poison.B -> 0)

(* The same, over one slot. *)
let fill_slice pool offset n poison =
  match pool with
  | Float32 a -> Array1.fill (Array1.sub a offset n) (float_poison poison)
  | Float64 a -> Array1.fill (Array1.sub a offset n) (float_poison poison)
  | Int16_unsigned a -> Array1.fill (Array1.sub a offset n) (int_poison poison)
  | Int32 a ->
      Array1.fill (Array1.sub a offset n) (Int32.of_int (int_poison poison))
  | Int64 a ->
      Array1.fill (Array1.sub a offset n) (Int64.of_int (int_poison poison))
  | Int8_unsigned a ->
      Array1.fill (Array1.sub a offset n)
        (match poison with Poison.A -> 1 | Poison.B -> 0)

let make_pool kind n =
  match kind with
  | Alloc_script.Kind.Float32 ->
      Some (Float32 (Array1.create float32 c_layout n))
  | Float64 -> Some (Float64 (Array1.create float64 c_layout n))
  | Int16_unsigned ->
      Some (Int16_unsigned (Array1.create int16_unsigned c_layout n))
  | Int32 -> Some (Int32 (Array1.create int32 c_layout n))
  | Int64 -> Some (Int64 (Array1.create int64 c_layout n))
  | Int8_unsigned ->
      Some (Int8_unsigned (Array1.create int8_unsigned c_layout n))
  | Int16_signed | Int8_signed -> None

let create ?poison plan =
  let alloc (p : Arena_plan.Pool.t) =
    let failed () =
      Err.fail ~pos:__POS__
        (`Arena_alloc_failed
           { Alloc_failed.kind = p.Arena_plan.Pool.kind; bytes = p.bytes })
    in
    (* A pool is bounded below [Hard.numel] by the plan, so the count fits an
       [int] on every backend. *)
    match
      make_pool p.Arena_plan.Pool.kind
        (Int64.to_int (Element_count.to_int64 p.numel))
    with
    | exception (Out_of_memory | Invalid_argument _) -> failed ()
    | None -> failed ()
    | Some pool -> Err.return (p.kind, pool)
  in
  let open Err.Syntax in
  let+ pools = Err.List.map alloc (Arena_plan.pools plan) in
  {
    plan;
    pools;
    poison;
    busy = false;
    copies = { Copies.count = 0L; bytes = Byte_size.zero };
  }

let footprint plan =
  let open Err.Syntax in
  let stats = Arena_plan.stats plan in
  let+ outside = Alloc_script.out_of_arena_bytes (Arena_plan.script plan) in
  (* Saturating: a footprint past [int64] is over every budget. *)
  match
    Err.payload (Byte_size.add stats.Arena_plan.Stats.pool_bytes outside)
  with
  | Ok total -> total
  | Error (`Quantity_overflow _) -> invariant (Byte_size.of_int64 Int64.max_int)

let view t id =
  match Arena_plan.slot t.plan id with
  | None -> Err.return None
  | Some slot -> (
      let bad () = Err.fail ~pos:__POS__ (`Arena_slot id) in
      match List.assoc_opt slot.Arena_plan.Slot.kind t.pools with
      | None -> bad ()
      | Some pool -> (
          let offset =
            invariant
              (Element_offset.of_bytes slot.Arena_plan.Slot.offset
                 (Alloc_script.Kind.element_bytes slot.Arena_plan.Slot.kind))
          in
          (* Inside a pool the plan bounded below [Hard.numel] cells, so both
             fit an [int] on every backend. *)
          let offset = Int64.to_int (Element_offset.to_int64 offset)
          and n =
            Int64.to_int (Element_count.to_int64 slot.Arena_plan.Slot.numel)
          in
          let shape = slot.Arena_plan.Slot.signature.Tensor_sig.shape in
          Option.iter (fun p -> fill_slice pool offset n p) t.poison;
          let real fmt data =
            Err.return
              (Some
                 (Tensor.Tensor
                    {
                      Tensor.shape;
                      payload = { Payload.fmt; quant = Payload.No_quant; data };
                    }))
          in
          let (Payload.Fmt fmt) =
            slot.Arena_plan.Slot.signature.Tensor_sig.fmt
          in
          match (pool, fmt) with
          | Float32 a, Payload.F32 -> real Payload.F32 (Array1.sub a offset n)
          | Float64 a, Payload.F64 -> real Payload.F64 (Array1.sub a offset n)
          | Int16_unsigned a, Payload.F16 ->
              real Payload.F16 (Array1.sub a offset n)
          | Int16_unsigned a, Payload.BF16 ->
              real Payload.BF16 (Array1.sub a offset n)
          | Int32 a, Payload.I32 -> real Payload.I32 (Array1.sub a offset n)
          | Int64 a, Payload.I64 -> real Payload.I64 (Array1.sub a offset n)
          | Int8_unsigned a, Payload.Bool ->
              real Payload.Bool (Array1.sub a offset n)
          | _ -> bad ()))

let settle t id ~dst ~result =
  if result == dst then Err.return dst
  else
    let open Err.Syntax in
    let+ () = Tensor.blit_into result dst in
    (match Arena_plan.slot t.plan id with
    | Some slot -> record_copy t ~bytes:slot.Arena_plan.Slot.bytes
    | None -> ());
    dst

let busy t = t.busy

let acquire t =
  if t.busy then Err.fail ~pos:__POS__ `Arena_busy
  else begin
    t.busy <- true;
    Err.return ()
  end

let release t = t.busy <- false

let with_run t f =
  match acquire t with
  | Error _ as e -> e
  | Ok () -> Fun.protect ~finally:(fun () -> release t) f

let fill_for_test t poison =
  List.iter (fun (_, pool) -> fill pool poison) t.pools
