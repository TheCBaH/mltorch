module Reason = Ssa_vector_body.Reason

module Decision = struct
  type outcome = Vectorized | Kept_scalar of Reason.t

  type t = {
    loop : Ssa_id.Region.t;
    trips : int64;
    work : int64;
    executions : int64;
    ops : (Ssa_target.Op.t * int) list;
    outcome : outcome;
  }
end

type report = Decision.t list

open Ssa_vector_body

(* ---- one loop ------------------------------------------------------------------ *)

(* Attempts one loop whose body holds no vector loop. *)
let attempt ~policy ~(target : Ssa_target.t) info t
    (s : Ssa_region.t Ssa_stmt.t) =
  match s with
  | Ssa_stmt.For f -> (
      let body_id = f.body.Ssa_region.id in
      let trips = ref 0L and ops = ref [] in
      let result =
        try
          if not (Int64.equal f.step 1L) then refuse Reason.Strided_loop;
          (match (f.body.Ssa_region.params, f.inits) with
          | [ _; _ ], [ _ ] -> ()
          | _ -> refuse Reason.Carries_values);
          let n =
            match
              Ssa_range.trips info.ranges ~lo:f.lo ~hi:f.hi ~step:f.step
            with
            | Ssa_range.Exactly n -> n
            | Ssa_range.Zero | Ssa_range.At_least_one | Ssa_range.Unknown ->
                refuse Reason.Non_constant_bounds
          in
          trips := n;
          let lo_c, hi_c =
            match
              ( Ssa_range.range info.ranges f.lo,
                Ssa_range.range info.ranges f.hi )
            with
            | Ssa_range.Range l, Ssa_range.Range h -> (l.lo, h.lo)
            | _ -> refuse Reason.Non_constant_bounds
          in
          let lanes = target.Ssa_target.lanes in
          let width = Ssa_type.Lanes.to_int lanes in
          if Int64.compare n (Int64.of_int width) < 0 then
            refuse (Reason.Too_short { trips = n; lanes });
          if (not target.Ssa_target.inner_loops) && has_loop f.body then
            refuse Reason.Inner_loops_declined;
          let iv = List.hd f.body.Ssa_region.params in
          let kinds = Hashtbl.create 64 in
          set kinds iv (Aff 1L);
          List.iter (classify_stmt kinds) f.body.Ssa_region.body;
          check_memory ~policy f.body.Ssa_region.body;
          let cx =
            make_ctx ~t ~program:info.program ~kinds ~consts:info.consts ~lanes
              ~trips:(fun id ->
                Option.value ~default:16
                  (Hashtbl.find_opt info.loop_trips (id :> int)))
          in
          let body, () =
            in_region cx (fun () ->
                List.iter (emit_stmt cx) f.body.Ssa_region.body)
          in
          ops := cx.ops;
          if not (Ssa_target.profitable target cx.ops) then
            refuse Reason.Unprofitable;
          (* the full groups, then the original loop for what is left *)
          let groups = Int64.div n (Int64.of_int width) in
          let full_hi =
            Int64.add lo_c (Int64.mul groups (Int64.of_int width))
          in
          ignore hi_c;
          let hi_v = Ssa_rewrite.fresh t (Ssa_type.Scalar Ssa_type.Index) in
          let hi_i =
            Ssa_stmt.Instr
              {
                Ssa_instr.results = [ hi_v ];
                op = Ssa_op.Const (Ssa_const.Index full_hi);
                token = None;
                origin = Ssa_origin.Unknown;
              }
          in
          let tail = not (Int64.equal full_hi hi_c) in
          let eff_v =
            if tail then Ssa_rewrite.fresh t Ssa_type.Effect
            else List.hd f.results
          in
          let vector_loop =
            Ssa_stmt.For
              {
                lo = f.lo;
                hi = hi_v;
                step = Int64.of_int width;
                inits = f.inits;
                results = [ eff_v ];
                body = { f.body with Ssa_region.body };
              }
          in
          let tail_stmts =
            if not tail then []
            else
              let cl = Ssa_clone.create t ~subst:[] in
              [
                Ssa_stmt.For
                  {
                    f with
                    lo = hi_v;
                    inits = [ eff_v ];
                    body = Ssa_clone.region cl f.body;
                  };
              ]
          in
          Ok (hi_i :: vector_loop :: tail_stmts)
        with Refuse r -> Error r
      in
      let n_ops = !ops in
      let decision outcome =
        {
          Decision.loop = body_id;
          trips = !trips;
          work = work info s;
          executions =
            Option.value ~default:1L
              (Hashtbl.find_opt info.executions (body_id :> int));
          ops = n_ops;
          outcome;
        }
      in
      match result with
      | Ok stmts ->
          Ssa_rewrite.mark_changed t;
          (stmts, decision Decision.Vectorized)
      | Error r -> ([ s ], decision (Decision.Kept_scalar r)))
  | Ssa_stmt.Instr _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _ ->
      invalid_arg "Ssa_vectorize.attempt: not a loop"

let program ?(alias = Ssa_effects.Conservative) ~target (p : Ssa_program.t) =
  let info = survey p in
  let decisions = ref [] in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.For { body; _ } when not (has_vector body) ->
        let stmts, decision = attempt ~policy:alias ~target info t s in
        decisions := decision :: !decisions;
        stmts
    | Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Instr _ | Ssa_stmt.Ordered_sum _
      ->
        [ s ]
  in
  let q, _ = Ssa_rewrite.program rule p in
  (match Err.payload (Ssa_verify.check q) with
  | Ok () -> ()
  | Error e ->
      invalid_arg
        (Fmt.str "Ssa_vectorize returned a program that does not verify: %a"
           Ssa_verify.pp_error e));
  (q, List.rev !decisions)

let tally (r : report) =
  let add acc name n w =
    let c, x = Option.value ~default:(0, 0L) (List.assoc_opt name acc) in
    (name, (c + n, Int64.add x w)) :: List.remove_assoc name acc
  in
  List.fold_left
    (fun acc (d : Decision.t) ->
      let weight = Int64.mul d.Decision.trips d.Decision.executions in
      let name =
        match d.Decision.outcome with
        | Decision.Vectorized -> "vectorized"
        | Decision.Kept_scalar r -> Reason.name r
      in
      add acc name 1 weight)
    [] r
  |> List.sort (fun (a, _) (b, _) -> compare a b)
