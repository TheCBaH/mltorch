open Ssa_vector_body

module Reason = struct
  type t =
    | Body_statement
    | Contiguous_load_missing
    | Lanes_declined
    | Non_constant_bounds
    | Term_uniform
    | Too_short of { trips : int64; lanes : Ssa_type.Lanes.t }
    | Unsupported of Ssa_vector_body.Reason.t

  let name = function
    | Body_statement -> "body_statement"
    | Contiguous_load_missing -> "contiguous_load_missing"
    | Lanes_declined -> "lanes_declined"
    | Non_constant_bounds -> "non_constant_bounds"
    | Term_uniform -> "term_uniform"
    | Too_short _ -> "too_short"
    | Unsupported r -> "unsupported:" ^ Ssa_vector_body.Reason.name r
end

module Decision = struct
  type outcome = Kept_sequential of Reason.t | Scheduled of { parts : int }
  type t = { loop : Ssa_id.Region.t; trips : int64; outcome : outcome }
end

type report = Decision.t list

(* At least this many full vectors, so the horizontal combine is a small part of
   the sum. *)
let min_vectors = 4
let max_parts = 4

exception Kept of Reason.t

let keep r = raise (Kept r)

let instr ?token ~results op =
  Ssa_stmt.Instr { Ssa_instr.results; op; token; origin = Ssa_origin.Unknown }

(* The adjacent-pair tree: neighbours are added, an odd one out carried on. *)
let rec tree add = function
  | [] -> invalid_arg "Ssa_vector_sum.tree"
  | [ x ] -> x
  | xs ->
      let rec pairs = function
        | a :: b :: rest ->
            let s = add a b in
            s :: pairs rest
        | rest -> rest
      in
      tree add (pairs xs)

let elt_type (v : Ssa_value.t) =
  match v.Ssa_value.ty with
  | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64) -> v.Ssa_value.ty
  | _ -> keep Reason.Term_uniform

let schedule ~(target : Ssa_target.t) info t (s : Ssa_region.t Ssa_stmt.t) =
  match s with
  | Ssa_stmt.Ordered_sum f ->
      let elt = elt_type f.seed in
      let n =
        match Ssa_range.trips info.ranges ~lo:f.lo ~hi:f.hi ~step:1L with
        | Ssa_range.Exactly n -> n
        | Ssa_range.Zero | Ssa_range.At_least_one | Ssa_range.Unknown ->
            keep Reason.Non_constant_bounds
      in
      let lo_c =
        match Ssa_range.range info.ranges f.lo with
        | Ssa_range.Range r -> r.lo
        | Ssa_range.Empty -> keep Reason.Non_constant_bounds
      in
      let lanes = target.Ssa_target.lanes in
      let width = Ssa_type.Lanes.to_int lanes in
      if width < 2 then keep Reason.Lanes_declined;
      if
        Int64.compare
          (Int64.div n (Int64.of_int width))
          (Int64.of_int min_vectors)
        < 0
      then keep (Reason.Too_short { trips = n; lanes });
      if
        List.exists
          (function Ssa_stmt.Instr _ -> false | _ -> true)
          f.body.Ssa_region.body
      then keep Reason.Body_statement;
      let iv, eff_param =
        match f.body.Ssa_region.params with
        | [ iv; e ] -> (iv, e)
        | _ -> invalid_arg "Ssa_vector_sum: a sum's parameters"
      in
      let term, chain =
        match f.body.Ssa_region.yields with
        | [ term; chain ] -> (term, chain)
        | _ -> invalid_arg "Ssa_vector_sum: a sum's yields"
      in
      let kinds = Hashtbl.create 32 in
      let cx =
        make_ctx ~t ~program:info.program ~kinds ~consts:info.consts ~lanes
          ~trips:(fun _ -> 16)
      in
      let idx_ty = Ssa_type.Scalar Ssa_type.Index in
      let const_stmt ty c =
        let v = Ssa_rewrite.fresh t ty in
        (v, instr ~results:[ v ] (Ssa_op.Const c))
      in
      (* one part's term: the body cloned at a base index, vectorized, with its
         effect threaded on from [eff] *)
      let part ~base ~eff =
        Hashtbl.replace kinds (base.Ssa_value.id :> int) (Aff 1L);
        let cl = Ssa_clone.create t ~subst:[ (iv, base); (eff_param, eff) ] in
        let stmts = Ssa_clone.stmts cl f.body.Ssa_region.body in
        let term' = Ssa_clone.value cl term
        and chain' = Ssa_clone.value cl chain in
        (try List.iter (classify_stmt kinds) stmts
         with Refuse r -> keep (Reason.Unsupported r));
        if kind_of kinds term' <> Vec then keep Reason.Term_uniform;
        let vec =
          try
            let out, v =
              in_region cx (fun () ->
                  List.iter (emit_stmt cx) stmts;
                  vec_of cx term')
            in
            (out, v)
          with Refuse r -> keep (Reason.Unsupported r)
        in
        (fst vec, snd vec, chain')
      in
      (* a dry run decides whether the term reads consecutive cells at all *)
      let probe_base, _ = const_stmt idx_ty (Ssa_const.Index lo_c) in
      ignore (part ~base:probe_base ~eff:eff_param);
      if not (List.mem_assoc Ssa_target.Op.Contiguous_load cx.ops) then
        keep Reason.Contiguous_load_missing;
      cx.ops <- [];
      let full = Int64.to_int (Int64.div n (Int64.of_int width)) in
      let parts = Stdlib.min max_parts (Stdlib.max 1 (full / 2)) in
      let rounds = full / parts and extra = full mod parts in
      let tail = Int64.to_int n - (full * width) in
      let vec_ty =
        Ssa_type.Vec
          ((match elt with Ssa_type.Scalar s -> s | _ -> Ssa_type.F64), lanes)
      in
      let out = ref [] in
      let emit s = out := s :: !out in
      let zero_scalar () =
        let c =
          match elt with
          | Ssa_type.Scalar Ssa_type.F32 -> Ssa_const.F32 0.
          | _ -> Ssa_const.F64 0.
        in
        let v, i = const_stmt elt c in
        emit i;
        v
      in
      let add_vec a b =
        let r = Ssa_rewrite.fresh t vec_ty in
        emit
          (instr ~results:[ r ]
             (Ssa_op.Lanewise (Ssa_op.Float_binary (Expr.Value.Add, a, b))));
        r
      in
      let add_scalar a b =
        let r = Ssa_rewrite.fresh t elt in
        emit (instr ~results:[ r ] (Ssa_op.Float_binary (Expr.Value.Add, a, b)));
        r
      in
      let zero_vec () =
        let z = zero_scalar () in
        let r = Ssa_rewrite.fresh t vec_ty in
        emit (instr ~results:[ r ] (Ssa_op.Vec_splat { element = z; lanes }));
        r
      in
      let accs = Array.init parts (fun _ -> zero_vec ()) in
      let cur_eff = ref f.token in
      (* the rounds, as a loop over groups of [parts] vectors *)
      if rounds > 0 then (
        let zero_i, zero_s = const_stmt idx_ty (Ssa_const.Index 0L) in
        let hi_i, hi_s =
          const_stmt idx_ty (Ssa_const.Index (Int64.of_int rounds))
        in
        emit hi_s;
        emit zero_s;
        let i = Ssa_rewrite.fresh t idx_ty in
        let eff_in = Ssa_rewrite.fresh t Ssa_type.Effect in
        let acc_in = Array.init parts (fun _ -> Ssa_rewrite.fresh t vec_ty) in
        let saved = !out in
        out := [];
        let stride = Int64.of_int (parts * width) in
        let scaled = Ssa_rewrite.fresh t idx_ty in
        emit
          (instr ~results:[ scaled ] (Ssa_op.Index_scale_in_domain (stride, i)));
        let eff = ref eff_in in
        let next =
          Array.mapi
            (fun j acc ->
              let c, c_s =
                const_stmt idx_ty
                  (Ssa_const.Index (Int64.add lo_c (Int64.of_int (j * width))))
              in
              emit c_s;
              let base = Ssa_rewrite.fresh t idx_ty in
              emit
                (instr ~results:[ base ]
                   (Ssa_op.Index_add_in_domain (scaled, c)));
              let body, term_vec, e = part ~base ~eff:!eff in
              List.iter emit body;
              eff := e;
              add_vec acc term_vec)
            acc_in
        in
        let loop_body = List.rev !out in
        out := saved;
        let results = Array.init parts (fun _ -> Ssa_rewrite.fresh t vec_ty) in
        let eff_out = Ssa_rewrite.fresh t Ssa_type.Effect in
        emit
          (Ssa_stmt.For
             {
               lo = zero_i;
               hi = hi_i;
               step = 1L;
               inits = Array.to_list accs @ [ !cur_eff ];
               results = Array.to_list results @ [ eff_out ];
               body =
                 {
                   Ssa_region.id =
                     (let id, nx =
                        Ssa_id.Region.Next.alloc t.Ssa_rewrite.next_region
                      in
                      t.Ssa_rewrite.next_region <- nx;
                      id);
                   params = (i :: Array.to_list acc_in) @ [ eff_in ];
                   body = loop_body;
                   yields = Array.to_list next @ [ !eff ];
                 };
             });
        Array.blit results 0 accs 0 parts;
        cur_eff := eff_out);
      (* the leftover vectors, straight-line, into the first accumulators *)
      for e = 0 to extra - 1 do
        let base_c, base_s =
          const_stmt idx_ty
            (Ssa_const.Index
               (Int64.add lo_c (Int64.of_int (((rounds * parts) + e) * width))))
        in
        emit base_s;
        let body, term_vec, eff' = part ~base:base_c ~eff:!cur_eff in
        List.iter emit body;
        cur_eff := eff';
        accs.(e) <- add_vec accs.(e) term_vec
      done;
      let combined = tree add_vec (Array.to_list accs) in
      let lane_values =
        List.init width (fun k ->
            let r = Ssa_rewrite.fresh t elt in
            emit
              (instr ~results:[ r ]
                 (Ssa_op.Vec_extract
                    { lane = Ssa_type.Lane.of_int k; vector = combined }));
            r)
      in
      let horizontal = tree add_scalar lane_values in
      let tail_sum =
        if tail = 0 then zero_scalar ()
        else
          let lo_v, lo_s =
            const_stmt idx_ty
              (Ssa_const.Index (Int64.add lo_c (Int64.of_int (full * width))))
          in
          emit lo_s;
          let seed = zero_scalar () in
          let cl = Ssa_clone.create t ~subst:[] in
          let sum = Ssa_rewrite.fresh t elt in
          let eff_out = Ssa_rewrite.fresh t Ssa_type.Effect in
          emit
            (Ssa_stmt.Ordered_sum
               {
                 lo = lo_v;
                 hi = f.hi;
                 seed;
                 token = !cur_eff;
                 results = [ sum; eff_out ];
                 body = Ssa_clone.region cl f.body;
               });
          cur_eff := eff_out;
          sum
      in
      let inner = add_scalar horizontal tail_sum in
      let result = add_scalar f.seed inner in
      Ssa_rewrite.alias t ~from:(List.hd f.results) ~to_:result;
      Ssa_rewrite.alias t ~from:(List.nth f.results 1) ~to_:!cur_eff;
      (parts, List.rev !out)
  | Ssa_stmt.Instr _ | Ssa_stmt.For _ | Ssa_stmt.If _ ->
      invalid_arg "Ssa_vector_sum.schedule: not a sum"

(* The sums nested under a vector loop: the loop's own lanes already took what
   they could. *)
let under_vector_loops (p : Ssa_program.t) =
  let inside = Hashtbl.create 16 in
  let rec region ~under (r : Ssa_region.t) =
    List.iter (stmt ~under) r.Ssa_region.body
  and stmt ~under : Ssa_region.t Ssa_stmt.t -> unit = function
    | Ssa_stmt.Instr _ -> ()
    | Ssa_stmt.Ordered_sum { body; _ } ->
        if under then Hashtbl.replace inside (body.Ssa_region.id :> int) ();
        region ~under body
    | Ssa_stmt.For { body; _ } -> region ~under:(under || has_vector body) body
    | Ssa_stmt.If { then_; else_; _ } ->
        region ~under then_;
        region ~under else_
  in
  region ~under:false p.Ssa_program.entry;
  inside

let program ~target (p : Ssa_program.t) =
  let info = survey p in
  let inside = under_vector_loops p in
  let decisions = ref [] in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Ordered_sum { body; seed; _ }
      when (not (Hashtbl.mem inside (body.Ssa_region.id :> int)))
           &&
           match seed.Ssa_value.ty with
           | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64) -> true
           | _ -> false -> (
        let trips =
          match s with
          | Ssa_stmt.Ordered_sum { lo; hi; _ } -> (
              match Ssa_range.trips info.ranges ~lo ~hi ~step:1L with
              | Ssa_range.Exactly n -> n
              | Ssa_range.Zero | Ssa_range.At_least_one | Ssa_range.Unknown ->
                  0L)
          | _ -> 0L
        in
        let decide outcome =
          decisions :=
            { Decision.loop = body.Ssa_region.id; trips; outcome } :: !decisions
        in
        match schedule ~target info t s with
        | parts, stmts ->
            Ssa_rewrite.mark_changed t;
            decide (Decision.Scheduled { parts });
            stmts
        | exception Kept r ->
            decide (Decision.Kept_sequential r);
            [ s ])
    | Ssa_stmt.Ordered_sum _ | Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Instr _
      ->
        [ s ]
  in
  let q, _ = Ssa_rewrite.program rule p in
  (match Err.payload (Ssa_verify.check q) with
  | Ok () -> ()
  | Error e ->
      invalid_arg
        (Fmt.str "Ssa_vector_sum returned a program that does not verify: %a"
           Ssa_verify.pp_error e));
  (q, List.rev !decisions)
