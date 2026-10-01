(* See arena_plan.mli. *)

open Core.Storage_units
module Kind = Alloc_script.Kind

(* Conversions ruled out by construction: a pool's size is a whole number of
   cells, since it ends where some block does, every block is a whole number of
   cells, and every block starts at an alignment of at least its cell width;
   the plan's limits are validated non-negative. *)
let invariant r = Err.or_raise ~pp_error r

module Slot = struct
  type t = {
    id : Tensor_id.t;
    signature : Tensor_sig.t;
    kind : Kind.t;
    offset : Byte_offset.t;
    bytes : Byte_size.t;
    numel : Element_count.t;
  }
end

module Pool = struct
  type t = {
    kind : Kind.t;
    bytes : Byte_size.t;
    numel : Element_count.t;
    alignment : Byte_alignment.t;
  }
end

module Stats = struct
  type t = {
    kinds : (Kind.t * Interval_alloc.Stats.t) list;
    pool_bytes : Byte_size.t;
    per_kind_bound_bytes : Byte_size.t;
    combined_bound_bytes : Byte_size.t;
    out_of_arena_bytes : Byte_size.t;
  }

  let pp ppf t =
    Format.fprintf ppf
      "@[<v>(for this node order)@,\
       pool %a bytes, per-kind bound %a, combined bound %a, outside the arena \
       %a"
      Byte_size.pp t.pool_bytes Byte_size.pp t.per_kind_bound_bytes Byte_size.pp
      t.combined_bound_bytes Byte_size.pp t.out_of_arena_bytes;
    List.iter
      (fun (kind, stats) ->
        Format.fprintf ppf "@,%a: %a" Kind.pp kind Interval_alloc.Stats.pp stats)
      t.kinds;
    Format.fprintf ppf "@]"
end

module Over_limit = struct
  type limit = Bytes of Byte_size.t | Elements of Element_count.t

  type t = {
    kind : Kind.t;
    numel : Element_count.t;
    bytes : Byte_size.t;
    limit : limit;
  }
end

module Placement_error = struct
  type t =
    [ `Duplicate_placement of Tensor_id.t
    | `Live_overflow of Tensor_id.t
    | `Misaligned of Tensor_id.t Interval_alloc.Misaligned.t
    | `Offset_overflow of Tensor_id.t
    | `Out_of_pool of Tensor_id.t Interval_alloc.Out_of_pool.t
    | `Overlap of Tensor_id.t Interval_alloc.Overlap.t
    | `Pool_overflow of Tensor_id.t
    | `Unknown_key of Tensor_id.t
    | `Unplaced of Tensor_id.t ]
end

type error =
  [ `Arena_over_limit of Over_limit.t
  | `Arena_placement of Placement_error.t
  | `Arena_script of Tensor_id.t ]

type t = {
  script : Alloc_script.t;
  policy : Alignment_policy.t;
  slots : Slot.t Tensor_id.Map.t;
  pools : Pool.t list;
  stats : Stats.t;
}

let script t = t.script
let policy t = t.policy
let slot t id = Tensor_id.Map.find_opt id t.slots
let slots t = List.map snd (Tensor_id.Map.bindings t.slots)
let pools t = t.pools
let stats t = t.stats

let base_alignment t =
  List.fold_left
    (fun acc (p : Pool.t) ->
      Some
        (Option.fold ~none:p.alignment
           ~some:(Byte_alignment.max p.alignment)
           acc))
    None t.pools

(* Sums stay far below [int64] (a pool is bounded by [Hard.numel] cells), but a
   sum of bounds is a different aggregate, so it is checked all the same. *)
let sum_checked id values =
  Err.List.fold_left
    (fun acc v ->
      Byte_size.add acc v
      |> Err.map_error ~pos:__POS__ (fun (`Quantity_overflow _) ->
          `Arena_placement (`Live_overflow id)))
    Byte_size.zero values

let place ?budget
    { Arena_problem.Kind_problem.kind = _; script = exact; padded } =
  let open Err.Syntax in
  let placement e = `Arena_placement (e : Placement_error.t) in
  (* A portfolio run on [script], its placement checked against the exact
     script and cut to its actual length there. *)
  let solve script =
    let* solution, stats =
      Interval_alloc.solve_best ?budget script
      |> Err.map_error ~pos:__POS__ (function
          | (`Live_overflow _ | `Pool_overflow _) as e -> placement e)
    in
    let+ witness =
      Interval_alloc.check exact solution
      |> Err.map_error ~pos:__POS__ (fun e -> placement e)
    in
    (Interval_alloc.fit exact witness, stats)
  in
  let* exact_witness, exact_stats = solve exact in
  let+ padded_witness, padded_stats = solve padded in
  let witness, stats =
    if
      Byte_size.compare
        (Interval_alloc.pool padded_witness)
        (Interval_alloc.pool exact_witness)
      < 0
    then (padded_witness, padded_stats)
    else (exact_witness, exact_stats)
  in
  ( witness,
    {
      stats with
      Interval_alloc.Stats.pool = Interval_alloc.pool witness;
      lower_bound = exact_stats.Interval_alloc.Stats.lower_bound;
    } )

let create ?(limits = Kernel.Limits.default) ?budget
    ?alignment:(policy = Alignment_policy.standard) (script : Alloc_script.t) =
  let open Err.Syntax in
  let* problem = Arena_problem.of_script script in
  let first_id = Arena_problem.first_id problem in
  let plan_kind ({ Arena_problem.Kind_problem.kind; padded; _ } as problem) =
    let placement e = `Arena_placement (e : Placement_error.t) in
    let* witness, stats = place ?budget problem in
    let* padded_bound =
      Interval_alloc.lower_bound padded
      |> Err.map_error ~pos:__POS__ (fun e -> placement e)
    in
    let bytes = Interval_alloc.pool witness in
    let numel =
      invariant (Element_count.of_bytes bytes (Kind.element_bytes kind))
    in
    let hard = invariant (Element_count.of_int64 Kernel.Limits.Hard.numel)
    and max_bytes =
      invariant (Byte_size.of_int64 limits.Kernel.Limits.max_bytes)
    in
    let over limit =
      Err.fail ~pos:__POS__
        (`Arena_over_limit { Over_limit.kind; numel; bytes; limit })
    in
    if Element_count.compare numel hard >= 0 then
      over (Over_limit.Elements hard)
    else if Byte_size.compare bytes max_bytes > 0 then
      over (Over_limit.Bytes max_bytes)
    else Err.return (kind, witness, stats, numel, bytes, padded_bound)
  in
  let* planned = Err.List.map plan_kind (Arena_problem.kinds problem) in
  let slots =
    List.fold_left
      (fun acc (kind, witness, _, _, _, _) ->
        List.fold_left
          (fun acc (id, offset) ->
            let a =
              match Arena_problem.eligible problem id with
              | Some a -> a
              | None -> assert false
            in
            Tensor_id.Map.add id
              {
                Slot.id;
                signature = a.Alloc_script.Alloc.signature;
                kind;
                offset;
                bytes = a.Alloc_script.Alloc.bytes;
                numel = a.Alloc_script.Alloc.numel;
              }
              acc)
          acc
          (Interval_alloc.placements witness))
      Tensor_id.Map.empty planned
  in
  (* A pool's base must be aligned as strictly as any slot in it. *)
  let alignment kind =
    Tensor_id.Map.fold
      (fun _ (s : Slot.t) acc ->
        if Kind.equal s.kind kind then
          match Arena_problem.eligible problem s.id with
          | Some a -> Byte_alignment.max acc a.Alloc_script.Alloc.alignment
          | None -> acc
        else acc)
      slots
      (invariant (Byte_alignment.of_element_bytes (Kind.element_bytes kind)))
  in
  let pools =
    List.map
      (fun (kind, _, _, numel, bytes, _) ->
        { Pool.kind; bytes; numel; alignment = alignment kind })
      planned
  in
  let* pool_bytes =
    sum_checked first_id (List.map (fun (p : Pool.t) -> p.Pool.bytes) pools)
  in
  let* per_kind_bound_bytes =
    sum_checked first_id
      (List.map (fun (_, _, _, _, _, padded_bound) -> padded_bound) planned)
  in
  let* combined_bound_bytes =
    Arena_problem.combined_bound_bytes problem
    |> Err.map_error ~pos:__POS__ (fun (`Peak_bytes_overflow id) ->
        `Arena_placement (`Live_overflow id))
  in
  let* out_of_arena_bytes =
    Arena_problem.out_of_arena_bytes problem
    |> Err.map_error ~pos:__POS__ (fun (`Peak_bytes_overflow id) ->
        `Arena_placement (`Live_overflow id))
  in
  Err.return
    {
      script;
      policy;
      slots;
      pools;
      stats =
        {
          Stats.kinds = List.map (fun (k, _, s, _, _, _) -> (k, s)) planned;
          pool_bytes;
          per_kind_bound_bytes;
          combined_bound_bytes;
          out_of_arena_bytes;
        };
    }

(* A slot's alignment under [policy]: what [Alloc_script.alloc] would record. *)
let slot_alignment policy (slot : Slot.t) =
  Alignment_policy.alignment policy slot.Slot.bytes
    ~payload_min:
      (invariant
         (Byte_alignment.of_element_bytes (Kind.element_bytes slot.Slot.kind)))

let revalidate t policy =
  let open Err.Syntax in
  (* Sizes, lifetimes and offsets are unchanged, so disjointness and bounds
     still hold: only each start's alignment is new. *)
  let* () =
    Err.List.iter
      (fun (slot : Slot.t) ->
        let alignment = slot_alignment policy slot in
        if Byte_offset.is_aligned slot.Slot.offset alignment then Err.return ()
        else
          Err.fail ~pos:__POS__
            (`Arena_placement
               (`Misaligned
                  {
                    Interval_alloc.Misaligned.block =
                      {
                        Interval_alloc.Block.key = slot.Slot.id;
                        offset = slot.Slot.offset;
                        size = slot.Slot.bytes;
                      };
                    alignment;
                  })))
      (slots t)
  in
  let realign = function
    | Alloc_script.Event.Alloc a ->
        Alloc_script.Event.Alloc
          {
            a with
            Alloc_script.Alloc.alignment =
              Alignment_policy.alignment policy a.Alloc_script.Alloc.bytes
                ~payload_min:
                  (invariant
                     (Byte_alignment.of_element_bytes
                        (Kind.element_bytes a.Alloc_script.Alloc.kind)));
          }
    | e -> e
  in
  let pool_alignment (p : Pool.t) =
    Tensor_id.Map.fold
      (fun _ (s : Slot.t) acc ->
        if Kind.equal s.kind p.kind then
          Byte_alignment.max acc (slot_alignment policy s)
        else acc)
      t.slots p.alignment
  in
  Err.return
    {
      t with
      script = List.map realign t.script;
      policy;
      pools =
        List.map
          (fun (p : Pool.t) -> { p with Pool.alignment = pool_alignment p })
          t.pools;
    }
