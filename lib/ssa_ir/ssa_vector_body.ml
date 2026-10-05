(* The body of a vector loop: what varies with the iteration, whether the loop's
   memory is independent across iterations, and the emission of one body's
   statements as vector operations. Shared by the loop vectorizer and the sum
   scheduler, which both turn a region of scalar statements into one whose
   values are vectors. *)

module Reason = struct
  type t =
    | Branch
    | Loop_carried of Ssa_id.Buffer.t
    | No_vector_form of Ssa_op.t
    | Aliasing of Ssa_id.Buffer.t
    | Carries_values
    | Inner_loops_declined
    | Non_affine_access
    | Non_constant_bounds
    | Store_through_broadcast
    | Strided_loop
    | Too_short of { trips : int64; lanes : Ssa_type.Lanes.t }
    | Unprofitable
    | Varying_bounds

  let name = function
    | Branch -> "branch"
    | Loop_carried _ -> "loop_carried"
    | No_vector_form op -> "no_vector_form:" ^ Ssa_op.name op
    | Aliasing _ -> "aliasing"
    | Carries_values -> "carries_values"
    | Inner_loops_declined -> "inner_loops_declined"
    | Non_affine_access -> "non_affine_access"
    | Non_constant_bounds -> "non_constant_bounds"
    | Store_through_broadcast -> "store_through_broadcast"
    | Strided_loop -> "strided_loop"
    | Too_short _ -> "too_short"
    | Unprofitable -> "unprofitable"
    | Varying_bounds -> "varying_bounds"

  let pp ppf r = Fmt.string ppf (name r)
end

exception Refuse of Reason.t

let refuse r = raise (Refuse r)

(* ---- what varies with the iteration ------------------------------------------

   Every value the loop's body defines is uniform (the same in every lane),
   affine in the induction value with a literal stride, or a vector. The
   induction value is affine with stride one. A value defined outside the loop is
   uniform and is not listed. *)

type kind = Uni | Aff of int64 | Vec

let kind_of kinds (v : Ssa_value.t) =
  match Hashtbl.find_opt kinds (v.Ssa_value.id :> int) with
  | Some k -> k
  | None -> Uni

let set kinds (v : Ssa_value.t) k =
  Hashtbl.replace kinds (v.Ssa_value.id :> int) k

let stride_ok s = Ssa_const.in_index_domain s

let add_kinds a b =
  match (a, b) with
  | Uni, Uni -> Uni
  | Aff s, Uni | Uni, Aff s -> Aff s
  | Aff s, Aff t ->
      let u = Int64.add s t in
      if not (stride_ok u) then refuse Reason.Non_affine_access
      else if Int64.equal u 0L then Uni
      else Aff u
  | Vec, _ | _, Vec -> refuse Reason.Non_affine_access

let scale_kind k = function
  | Uni -> Uni
  | Aff s ->
      let p = Int64.mul k s in
      if not (stride_ok p) then refuse Reason.Non_affine_access
      else if Int64.equal p 0L then Uni
      else Aff p
  | Vec -> refuse Reason.Non_affine_access

let all_uniform kinds op ~otherwise =
  if List.for_all (fun v -> kind_of kinds v = Uni) (Ssa_op.operands op) then Uni
  else refuse otherwise

let float_result (r : Ssa_value.t) =
  match r.Ssa_value.ty with
  | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64 | Ssa_type.Pred) -> true
  | _ -> false

let classify_instr kinds (i : Ssa_instr.t) =
  let op = i.Ssa_instr.op in
  let result () = List.hd i.Ssa_instr.results in
  let kind v = kind_of kinds v in
  match op with
  | Ssa_op.Const _ -> set kinds (result ()) Uni
  | Ssa_op.Index_add_in_domain (a, b) ->
      set kinds (result ()) (add_kinds (kind a) (kind b))
  | Ssa_op.Index_scale_in_domain (k, a) ->
      set kinds (result ()) (scale_kind k (kind a))
  | Ssa_op.Convert (Ssa_op.Convert.Index_to_f64, a) ->
      set kinds (result ())
        (match kind a with
        | Uni -> Uni
        | Aff _ -> Vec
        | Vec -> refuse Reason.Non_affine_access)
  | Ssa_op.Convert ((Ssa_op.Convert.F32_to_f64 | Ssa_op.Convert.F64_to_f32), _)
  | Ssa_op.Float_binary _ | Ssa_op.Float_compare _ | Ssa_op.Float_fma _
  | Ssa_op.Float_max _ | Ssa_op.Float_unary _ | Ssa_op.Pool_better _
  | Ssa_op.Pred_not _ | Ssa_op.Pred_or _ | Ssa_op.Select _ ->
      let ks = List.map kind (Ssa_op.operands op) in
      if List.exists (function Aff _ -> true | Uni | Vec -> false) ks then
        refuse (Reason.No_vector_form op)
      else if List.mem Vec ks then
        if float_result (result ()) then set kinds (result ()) Vec
        else refuse (Reason.No_vector_form op)
      else set kinds (result ()) Uni
  | Ssa_op.Convert
      ( ( Ssa_op.Convert.Index_to_i64 | Ssa_op.Convert.I64_to_f64
        | Ssa_op.Convert.I64_to_f32 ),
        _ )
  | Ssa_op.I64_arith _ | Ssa_op.I64_compare _ ->
      set kinds (result ())
        (all_uniform kinds op ~otherwise:(Reason.No_vector_form op))
  | Ssa_op.Index_ceil_div _ | Ssa_op.Index_clamp_low _ | Ssa_op.Index_compare _
  | Ssa_op.Index_floor_div _ | Ssa_op.Index_max _ | Ssa_op.Index_min _ ->
      set kinds (result ())
        (all_uniform kinds op ~otherwise:Reason.Non_affine_access)
  | Ssa_op.Load_in_bounds { at; decode; _ } -> (
      match at with
      | Ssa_access.Flat v ->
          if kind v = Uni then set kinds (result ()) Uni
          else refuse Reason.Non_affine_access
      | Ssa_access.Coord c ->
          let ks = List.map kind (Expr.Coord.to_list c) in
          if List.mem Vec ks then refuse Reason.Non_affine_access
          else if List.for_all (fun k -> k = Uni) ks then
            set kinds (result ()) Uni
          else (
            (match decode with
            | Ssa_op.Decode.I64 -> refuse (Reason.No_vector_form op)
            | Ssa_op.Decode.Bf16_to_f64 | Ssa_op.Decode.Bool_to_f64
            | Ssa_op.Decode.F16_to_f64 | Ssa_op.Decode.F32_to_f64
            | Ssa_op.Decode.F64_to_f64 | Ssa_op.Decode.I16_dequant
            | Ssa_op.Decode.I32_to_f64 | Ssa_op.Decode.I64_to_f64
            | Ssa_op.Decode.I8_dequant ->
                ());
            set kinds (result ()) Vec))
  | Ssa_op.Mark _ | Ssa_op.Store _ -> ()
  | Ssa_op.Check_access _ | Ssa_op.Check_gather _ | Ssa_op.Check_local _
  | Ssa_op.Check_scan _ | Ssa_op.Float_to_i64 _ | Ssa_op.I64_div _
  | Ssa_op.Index_add _ | Ssa_op.Index_of_i64 _ | Ssa_op.Index_scale _
  | Ssa_op.Lanewise _ | Ssa_op.Load _ | Ssa_op.Local_alloc _
  | Ssa_op.Local_read _ | Ssa_op.Local_write _ | Ssa_op.Mark_lanes _
  | Ssa_op.Meter_charge | Ssa_op.Meter_release _ | Ssa_op.Meter_reserve _
  | Ssa_op.Meter_reset | Ssa_op.Vec_extract _ | Ssa_op.Vec_insert _
  | Ssa_op.Vec_iota _ | Ssa_op.Vec_load _ | Ssa_op.Vec_splat _
  | Ssa_op.Vec_store _ ->
      refuse (Reason.No_vector_form op)

(* The kind of a carried value: a vector once anything that can reach it is one.
   Iterated to a fixed point, because a yield that varies makes its parameter
   vary, which can make another yield vary in turn. *)
let rec classify_stmt kinds : Ssa_region.t Ssa_stmt.t -> unit = function
  | Ssa_stmt.Instr i -> classify_instr kinds i
  | Ssa_stmt.If _ -> refuse Reason.Branch
  | Ssa_stmt.For f -> (
      if kind_of kinds f.lo <> Uni || kind_of kinds f.hi <> Uni then
        refuse Reason.Varying_bounds;
      match f.body.Ssa_region.params with
      | [] -> invalid_arg "Ssa_vectorize: a loop without an induction value"
      | iv :: carried ->
          set kinds iv Uni;
          List.iter2
            (fun p init ->
              match kind_of kinds init with
              | Uni -> set kinds p Uni
              | Vec -> set kinds p Vec
              | Aff _ -> refuse Reason.Non_affine_access)
            carried f.inits;
          let rec fix () =
            List.iter (classify_stmt kinds) f.body.Ssa_region.body;
            let changed = ref false in
            List.iter2
              (fun p y ->
                match kind_of kinds y with
                | Vec ->
                    if kind_of kinds p <> Vec then (
                      set kinds p Vec;
                      changed := true)
                | Aff _ -> refuse Reason.Non_affine_access
                | Uni -> ())
              carried f.body.Ssa_region.yields;
            if !changed then fix ()
          in
          fix ();
          List.iter2
            (fun r p -> set kinds r (kind_of kinds p))
            f.results carried)
  | Ssa_stmt.Ordered_sum f ->
      if kind_of kinds f.lo <> Uni || kind_of kinds f.hi <> Uni then
        refuse Reason.Varying_bounds;
      (match f.body.Ssa_region.params with
      | iv :: _ -> set kinds iv Uni
      | [] -> invalid_arg "Ssa_vectorize: a sum without an induction value");
      List.iter (classify_stmt kinds) f.body.Ssa_region.body;
      let term = List.hd f.body.Ssa_region.yields in
      let k =
        match (kind_of kinds f.seed, kind_of kinds term) with
        | Uni, Uni -> Uni
        | (Uni | Vec), (Uni | Vec) -> Vec
        | Aff _, _ | _, Aff _ -> refuse Reason.Non_affine_access
      in
      set kinds (List.hd f.results) k

(* ---- memory ------------------------------------------------------------------

   Iterations are independent when each reads and writes only cells of its own.
   The check is deliberately narrow: a buffer the loop writes is accessed
   everywhere in it at one and the same coordinate, so a lane meets exactly the
   cell its scalar iteration did, and, unless the policy rules overlap out, it
   touches no other buffer. *)

type access = {
  buffer : Ssa_id.Buffer.t;
  coords : int list;  (** the value ids of the six coordinates *)
  write : bool;
}

let accesses body =
  let out = ref [] in
  let coords (c : Ssa_value.t Expr.Coord.t) =
    List.map
      (fun (v : Ssa_value.t) -> (v.Ssa_value.id :> int))
      (Expr.Coord.to_list c)
  in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt : Ssa_region.t Ssa_stmt.t -> unit = function
    | Ssa_stmt.Instr i -> (
        match i.Ssa_instr.op with
        | Ssa_op.Load_in_bounds { buffer; at = Ssa_access.Coord c; _ } ->
            out := { buffer; coords = coords c; write = false } :: !out
        | Ssa_op.Store { buffer; at = Ssa_access.Coord c; _ } ->
            out := { buffer; coords = coords c; write = true } :: !out
        | Ssa_op.Load_in_bounds { buffer; at = Ssa_access.Flat v; _ } ->
            out :=
              { buffer; coords = [ (v.Ssa_value.id :> int) ]; write = false }
              :: !out
        | Ssa_op.Store { at = Ssa_access.Flat _; _ } ->
            refuse Reason.Non_affine_access
        | _ -> ())
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If _ -> refuse Reason.Branch
  in
  List.iter stmt body;
  List.rev !out

let check_memory ~policy body =
  let all = accesses body in
  let stores = List.filter (fun a -> a.write) all in
  List.iter
    (fun (s : access) ->
      List.iter
        (fun (a : access) ->
          if Ssa_id.Buffer.equal a.buffer s.buffer then (
            if a.coords <> s.coords then refuse (Reason.Loop_carried s.buffer))
          else
            match policy with
            | Ssa_effects.Conservative -> refuse (Reason.Aliasing a.buffer)
            | Ssa_effects.Distinct_buffers -> ())
        all)
    stores

(* ---- emission ----------------------------------------------------------------- *)

type ctx = {
  t : Ssa_rewrite.t;
  program : Ssa_program.t;
  kinds : (int, kind) Hashtbl.t;
  consts : (int, unit) Hashtbl.t;
  lanes : Ssa_type.Lanes.t;
  vmap : (int, Ssa_value.t) Hashtbl.t;
  mutable splats : (int, Ssa_value.t) Hashtbl.t list;
  mutable out : Ssa_region.t Ssa_stmt.t list;
  mutable ops : (Ssa_target.Op.t * int) list;
  mutable weight : int;
  trips : Ssa_id.Region.t -> int;
      (** a loop body's iteration count, the cost model's weight *)
}

let count cx op =
  cx.ops <-
    (match List.assoc_opt op cx.ops with
    | Some n -> (op, n + cx.weight) :: List.remove_assoc op cx.ops
    | None -> (op, cx.weight) :: cx.ops)

let push cx s = cx.out <- s :: cx.out
let fresh cx ty = Ssa_rewrite.fresh cx.t ty

(* Runs [f] to build one region's statements, with its own splat scope. *)
let in_region cx f =
  let saved_out = cx.out and saved_splats = cx.splats in
  cx.out <- [];
  cx.splats <- Hashtbl.copy (List.hd saved_splats) :: saved_splats;
  Fun.protect
    ~finally:(fun () ->
      cx.out <- saved_out;
      cx.splats <- saved_splats)
    (fun () ->
      let r = f () in
      (List.rev cx.out, r))

let vector_type cx (ty : Ssa_type.t) =
  match Ssa_typing.vector_of cx.lanes ty with
  | Ok t -> t
  | Error _ -> refuse Reason.Non_affine_access

let vec_of cx (v : Ssa_value.t) =
  match kind_of cx.kinds v with
  | Vec -> (
      match Hashtbl.find_opt cx.vmap (v.Ssa_value.id :> int) with
      | Some w -> w
      | None -> invalid_arg "Ssa_vectorize: a vector with no definition")
  | Aff _ -> refuse Reason.Non_affine_access
  | Uni -> (
      count cx
        (if Hashtbl.mem cx.consts (v.Ssa_value.id :> int) then
           Ssa_target.Op.Const
         else Ssa_target.Op.Splat);
      let scope = List.hd cx.splats in
      match Hashtbl.find_opt scope (v.Ssa_value.id :> int) with
      | Some s -> s
      | None ->
          let s = fresh cx (vector_type cx v.Ssa_value.ty) in
          push cx
            (Ssa_stmt.Instr
               {
                 Ssa_instr.results = [ s ];
                 op = Ssa_op.Vec_splat { element = v; lanes = cx.lanes };
                 token = None;
                 origin = Ssa_origin.Unknown;
               });
          Hashtbl.replace scope (v.Ssa_value.id :> int) s;
          s)

(* The lane step along each axis of a coordinate: a uniform component does not
   move, an affine one moves by its stride. *)
let steps_of cx (c : Ssa_value.t Expr.Coord.t) =
  Expr.Coord.map
    (fun v ->
      match kind_of cx.kinds v with
      | Uni -> 0L
      | Aff s -> s
      | Vec -> refuse Reason.Non_affine_access)
    c

(* How a vector memory access reaches its cells: one element apart, or not. *)
let access_class cx buffer steps =
  match Ssa_program.find_buffer cx.program buffer with
  | None -> `Strided
  | Some b ->
      let stride = Ssa_memory.offset b.Ssa_buffer.extents steps in
      if Int64.equal stride 0L then `Broadcast
      else if Int64.equal stride 1L then `Contiguous
      else `Strided

let load_op cx buffer steps decode =
  match access_class cx buffer steps with
  | `Broadcast -> Ssa_target.Op.Broadcast_load
  | `Strided -> Ssa_target.Op.Strided_load
  | `Contiguous -> (
      match decode with
      | Ssa_op.Decode.I32_to_f64 -> Ssa_target.Op.Convert_i32_load
      | Ssa_op.Decode.Bf16_to_f64 | Ssa_op.Decode.Bool_to_f64
      | Ssa_op.Decode.F16_to_f64 | Ssa_op.Decode.F32_to_f64
      | Ssa_op.Decode.F64_to_f64 | Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I64
      | Ssa_op.Decode.I64_to_f64 | Ssa_op.Decode.I8_dequant ->
          Ssa_target.Op.Contiguous_load)

let store_op cx buffer steps encode =
  match encode with
  | Ssa_op.Encode.Bool_nonzero -> Ssa_target.Op.Bool_store
  | Ssa_op.Encode.F32_round | Ssa_op.Encode.I64 -> (
      match access_class cx buffer steps with
      | `Contiguous -> Ssa_target.Op.Contiguous_store
      | `Broadcast | `Strided -> Ssa_target.Op.Strided_store)

let lanewise_op (op : Ssa_op.t) =
  match op with
  | Ssa_op.Float_binary (Expr.Value.Add, _, _) -> Some Ssa_target.Op.Add
  | Ssa_op.Float_binary (Expr.Value.Div, _, _) -> Some Ssa_target.Op.Div
  | Ssa_op.Float_binary (Expr.Value.Mul, _, _) -> Some Ssa_target.Op.Mul
  | Ssa_op.Float_binary (Expr.Value.Sub, _, _) -> Some Ssa_target.Op.Sub
  | Ssa_op.Float_max _ -> Some Ssa_target.Op.Float_max
  | Ssa_op.Float_fma _ -> Some Ssa_target.Op.Fma
  | Ssa_op.Convert (Ssa_op.Convert.F64_to_f32, _) ->
      Some Ssa_target.Op.Round_f32
  | Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, _) -> None
  | Ssa_op.Float_unary ((Expr.Value.Sqrt | Expr.Value.Trunc), _) ->
      Some Ssa_target.Op.Sqrt_trunc
  | Ssa_op.Float_unary
      ( ( Expr.Value.Cos | Expr.Value.Erf | Expr.Value.Exp | Expr.Value.Log
        | Expr.Value.Sin ),
        _ ) ->
      Some Ssa_target.Op.Transcendental
  | Ssa_op.Float_compare _ | Ssa_op.Pool_better _ ->
      Some Ssa_target.Op.Value_compare
  | Ssa_op.Pred_not _ | Ssa_op.Pred_or _ -> Some Ssa_target.Op.Logic
  | Ssa_op.Select _ -> Some Ssa_target.Op.Select
  | _ -> None

let emit_instr cx (i : Ssa_instr.t) =
  let op = i.Ssa_instr.op in
  match op with
  | Ssa_op.Mark mark ->
      push cx
        (Ssa_stmt.Instr
           {
             i with
             Ssa_instr.op = Ssa_op.Mark_lanes { mark; lanes = cx.lanes };
           })
  | Ssa_op.Store { buffer; at = Ssa_access.Coord at; encode; value } ->
      let steps = steps_of cx at in
      if List.for_all (Int64.equal 0L) (Expr.Coord.to_list steps) then
        refuse Reason.Store_through_broadcast;
      let v = vec_of cx value in
      count cx (store_op cx buffer steps encode);
      push cx
        (Ssa_stmt.Instr
           {
             i with
             Ssa_instr.op =
               Ssa_op.Vec_store
                 { buffer; at; steps; encode; value = v; lanes = cx.lanes };
           })
  | _ -> (
      let result = List.hd i.Ssa_instr.results in
      match kind_of cx.kinds result with
      | Uni | Aff _ -> push cx (Ssa_stmt.Instr i)
      | Vec -> (
          let define ty = fresh cx ty in
          let finish ?token ~results op =
            push cx (Ssa_stmt.Instr { i with Ssa_instr.results; op; token })
          in
          match op with
          | Ssa_op.Convert (Ssa_op.Convert.Index_to_f64, a) -> (
              match kind_of cx.kinds a with
              | Aff step ->
                  let r = define (Ssa_type.Vec (Ssa_type.F64, cx.lanes)) in
                  Hashtbl.replace cx.vmap (result.Ssa_value.id :> int) r;
                  count cx Ssa_target.Op.Index_value;
                  finish ~results:[ r ]
                    (Ssa_op.Vec_iota { base = a; step; lanes = cx.lanes })
              | Uni | Vec -> refuse Reason.Non_affine_access)
          | Ssa_op.Load_in_bounds { buffer; at = Ssa_access.Coord at; decode }
            ->
              let steps = steps_of cx at in
              let r = define (Ssa_type.Vec (Ssa_type.F64, cx.lanes)) in
              Hashtbl.replace cx.vmap (result.Ssa_value.id :> int) r;
              count cx (load_op cx buffer steps decode);
              finish ?token:i.Ssa_instr.token
                ~results:[ r; List.nth i.Ssa_instr.results 1 ]
                (Ssa_op.Vec_load { buffer; at; steps; decode; lanes = cx.lanes })
          | _ -> (
              (match lanewise_op op with Some o -> count cx o | None -> ());
              let lifted = Ssa_op.map_operands (vec_of cx) op in
              let lanewise = Ssa_op.Lanewise lifted in
              match Ssa_typing.result_types lanewise with
              | Ok [ ty ] ->
                  let r = define ty in
                  Hashtbl.replace cx.vmap (result.Ssa_value.id :> int) r;
                  finish ~results:[ r ] lanewise
              | Ok _ | Error _ -> refuse (Reason.No_vector_form op))))

let rec emit_stmt cx : Ssa_region.t Ssa_stmt.t -> unit = function
  | Ssa_stmt.Instr i -> emit_instr cx i
  | Ssa_stmt.If _ -> refuse Reason.Branch
  | Ssa_stmt.For f ->
      let iv, carried =
        match f.body.Ssa_region.params with
        | iv :: carried -> (iv, carried)
        | [] -> invalid_arg "Ssa_vectorize: a loop without an induction value"
      in
      (* a carried value that varies becomes a vector parameter *)
      let new_params =
        List.map
          (fun (p : Ssa_value.t) ->
            if kind_of cx.kinds p = Vec then (
              let q = fresh cx (vector_type cx p.Ssa_value.ty) in
              Hashtbl.replace cx.vmap (p.Ssa_value.id :> int) q;
              q)
            else p)
          carried
      in
      let inits =
        List.map2
          (fun (p : Ssa_value.t) init ->
            if kind_of cx.kinds p = Vec then vec_of cx init else init)
          carried f.inits
      in
      let results =
        List.map2
          (fun (p : Ssa_value.t) (r : Ssa_value.t) ->
            if kind_of cx.kinds p = Vec then (
              let q = fresh cx (vector_type cx r.Ssa_value.ty) in
              Hashtbl.replace cx.vmap (r.Ssa_value.id :> int) q;
              q)
            else r)
          carried f.results
      in
      let saved = cx.weight in
      cx.weight <- cx.weight * cx.trips f.body.Ssa_region.id;
      let body, yields =
        in_region cx (fun () ->
            List.iter (emit_stmt cx) f.body.Ssa_region.body;
            List.map2
              (fun (p : Ssa_value.t) y ->
                if kind_of cx.kinds p = Vec then vec_of cx y else y)
              carried f.body.Ssa_region.yields)
      in
      cx.weight <- saved;
      push cx
        (Ssa_stmt.For
           {
             f with
             inits;
             results;
             body =
               {
                 Ssa_region.id = f.body.Ssa_region.id;
                 params = iv :: new_params;
                 body;
                 yields;
               };
           })
  | Ssa_stmt.Ordered_sum f ->
      let sum = List.hd f.results in
      let vector = kind_of cx.kinds sum = Vec in
      let seed = if vector then vec_of cx f.seed else f.seed in
      let saved = cx.weight in
      cx.weight <- cx.weight * cx.trips f.body.Ssa_region.id;
      let body, yields =
        in_region cx (fun () ->
            List.iter (emit_stmt cx) f.body.Ssa_region.body;
            match f.body.Ssa_region.yields with
            | [ term; chain ] ->
                [ (if vector then vec_of cx term else term); chain ]
            | _ -> invalid_arg "Ssa_vectorize: a sum's yields")
      in
      cx.weight <- saved;
      let results =
        if vector then (
          let q = fresh cx (vector_type cx sum.Ssa_value.ty) in
          Hashtbl.replace cx.vmap (sum.Ssa_value.id :> int) q;
          [ q; List.nth f.results 1 ])
        else f.results
      in
      push cx
        (Ssa_stmt.Ordered_sum
           {
             f with
             seed;
             results;
             body = { f.body with Ssa_region.body; yields };
           })

(* ---- the facts about a program every attempt starts from ---------------------- *)

let rec has_loop (r : Ssa_region.t) =
  List.exists
    (function
      | Ssa_stmt.For _ | Ssa_stmt.Ordered_sum _ -> true
      | Ssa_stmt.If { then_; else_; _ } -> has_loop then_ || has_loop else_
      | Ssa_stmt.Instr _ -> false)
    r.Ssa_region.body

let rec has_vector (r : Ssa_region.t) =
  List.exists
    (function
      | Ssa_stmt.Instr i -> (
          match i.Ssa_instr.op with
          | Ssa_op.Lanewise _ | Ssa_op.Mark_lanes _ | Ssa_op.Vec_extract _
          | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _ | Ssa_op.Vec_load _
          | Ssa_op.Vec_splat _ | Ssa_op.Vec_store _ ->
              true
          | _ -> false)
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          has_vector body
      | Ssa_stmt.If { then_; else_; _ } -> has_vector then_ || has_vector else_)
    r.Ssa_region.body

type info = {
  program : Ssa_program.t;
  ranges : Ssa_range.t;
  consts : (int, unit) Hashtbl.t;
  loop_trips : (int, int) Hashtbl.t;  (** body region id -> constant trips *)
  executions : (int, int64) Hashtbl.t;
      (** body region id -> enclosing product *)
}

let survey (p : Ssa_program.t) =
  let ranges = Ssa_range.analyze p in
  let consts = Hashtbl.create 64 in
  let loop_trips = Hashtbl.create 16 and executions = Hashtbl.create 16 in
  let rec region ~enclosing (r : Ssa_region.t) =
    List.iter (stmt ~enclosing) r.Ssa_region.body
  and stmt ~enclosing : Ssa_region.t Ssa_stmt.t -> unit = function
    | Ssa_stmt.Instr i -> (
        match i.Ssa_instr.op with
        | Ssa_op.Const _ ->
            Hashtbl.replace consts
              ((List.hd i.Ssa_instr.results).Ssa_value.id :> int)
              ()
        | _ -> ())
    | Ssa_stmt.For { lo; hi; step; body; _ } ->
        let n =
          match Ssa_range.trips ranges ~lo ~hi ~step with
          | Ssa_range.Exactly n -> Some n
          | Ssa_range.Zero | Ssa_range.At_least_one | Ssa_range.Unknown -> None
        in
        Hashtbl.replace executions (body.Ssa_region.id :> int) enclosing;
        (match n with
        | Some n ->
            Hashtbl.replace loop_trips
              (body.Ssa_region.id :> int)
              (Int64.to_int n)
        | None -> ());
        region
          ~enclosing:(Int64.mul enclosing (Option.value n ~default:1L))
          body
    | Ssa_stmt.Ordered_sum { lo; hi; body; _ } ->
        let n =
          match Ssa_range.trips ranges ~lo ~hi ~step:1L with
          | Ssa_range.Exactly n -> Some n
          | Ssa_range.Zero | Ssa_range.At_least_one | Ssa_range.Unknown -> None
        in
        Hashtbl.replace executions (body.Ssa_region.id :> int) enclosing;
        (match n with
        | Some n ->
            Hashtbl.replace loop_trips
              (body.Ssa_region.id :> int)
              (Int64.to_int n)
        | None -> ());
        region
          ~enclosing:(Int64.mul enclosing (Option.value n ~default:1L))
          body
    | Ssa_stmt.If { then_; else_; _ } ->
        region ~enclosing then_;
        region ~enclosing else_
  in
  region ~enclosing:1L p.Ssa_program.entry;
  { program = p; ranges; consts; loop_trips; executions }

(* The innermost-loop iterations one execution of a loop covers. *)
let rec work info (s : Ssa_region.t Ssa_stmt.t) =
  let trips (body : Ssa_region.t) =
    Int64.of_int
      (Option.value ~default:1
         (Hashtbl.find_opt info.loop_trips (body.Ssa_region.id :> int)))
  in
  let inner (body : Ssa_region.t) =
    let sum =
      List.fold_left
        (fun acc s ->
          match s with
          | Ssa_stmt.For _ | Ssa_stmt.Ordered_sum _ ->
              Int64.add acc (work info s)
          | Ssa_stmt.Instr _ | Ssa_stmt.If _ -> acc)
        0L body.Ssa_region.body
    in
    if Int64.equal sum 0L then 1L else sum
  in
  match s with
  | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
      Int64.mul (trips body) (inner body)
  | Ssa_stmt.If _ | Ssa_stmt.Instr _ -> 0L

let make_ctx ~t ~program ~kinds ~consts ~lanes ~trips =
  {
    t;
    program;
    kinds;
    consts;
    lanes;
    vmap = Hashtbl.create 32;
    splats = [ Hashtbl.create 8 ];
    out = [];
    ops = [];
    weight = 1;
    trips;
  }
