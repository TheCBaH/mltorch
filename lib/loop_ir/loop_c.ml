open Loop_c_base
open Loop_c_vector

let rec stmt nm ~limits ~depth (s : Loop_stmt.t) : string list =
  let ind = indent depth in
  let line l = [ ind ^ l ] in
  match s with
  | Loop_stmt.Alloc (a, n) ->
      let n = (n :> int) in
      let off = nm.local_doubles in
      let a = array nm a in
      if nm.f32 then (
        (* Binary32 cells pack two to a [double] of scratch. *)
        nm.local_doubles <- Int64.add off (Int64.of_int ((n + 1) / 2));
        line
          (Printf.sprintf
             "float *%s = (float *)(local + %Ld); memset(%s, 0, %d * \
              sizeof(float));"
             a off a n))
      else (
        nm.local_doubles <- Int64.add off (Int64.of_int n);
        line
          (Printf.sprintf
             "double *%s = local + %Ld; memset(%s, 0, %d * sizeof(double));" a
             off a n))
  | Loop_stmt.Array_set (a, i, e) ->
      let i = index nm i in
      let e = num nm e in
      line (Printf.sprintf "%s[%s] = %s;" (array nm a) i e)
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      line (Printf.sprintf "%s = %s;" (temp nm t) (num nm e))
  | Loop_stmt.Assign (Loop_carrier.Int64, t, e) ->
      line (Printf.sprintf "%s = %s;" (temp nm t) (big nm e))
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      line (Printf.sprintf "%s = %s;" (index_temp nm t) (big nm e))
  | Loop_stmt.Assign_index (t, i) ->
      line (Printf.sprintf "%s = %s;" (index_temp nm t) (index nm i))
  | Loop_stmt.Fail_if (p, f) -> (
      let site = next_site nm f in
      match (p, f) with
      | Loop_bool.Index_overflows i, Loop_failure.Index_overflow { index = j }
        when i = j ->
          List.map
            (fun n ->
              ind
              ^ Printf.sprintf "if %s %s" (outside_int32 n.value)
                  (fail_record F.Kind.Index_overflow
                     [ string_of_int n.op; n.lhs; n.rhs ]))
            (overflow_nodes nm i)
      | _, Loop_failure.Index_overflow _ ->
          invalid_arg
            "Loop_c: an index overflow failure under a foreign predicate"
      | _ ->
          let p = pred nm p in
          line (Printf.sprintf "if %s %s" p (failure nm ~site f)))
  | Loop_stmt.For { var = v; lo; hi = hi_ix; body } ->
      let name = var nm v in
      let lo = index nm lo in
      let hi = index nm hi_ix in
      (* The interpreter evaluates [hi] once, on entry; a C [for] test runs per
         iteration. A literal or a loop variable cannot change meanwhile, so it
         stays in the test; anything else is bound once before the loop. *)
      let inline, limit =
        match hi_ix with
        | Loop_index.Const _ | Loop_index.Var _ -> ([], hi)
        | _ ->
            let n = bound nm v in
            ([ ind ^ "  const int64_t " ^ n ^ " = " ^ hi ^ ";" ], n)
      in
      [ ind ^ "{" ]
      @ inline
      @ [
          Printf.sprintf "%s  for (int64_t %s = %s; %s < %s; %s++) {" ind name
            lo name limit name;
        ]
      @ block nm ~limits ~depth:(depth + 2) body
      @ [ ind ^ "  }"; ind ^ "}" ]
  | Loop_stmt.If (p, yes, no) ->
      let p = pred nm p in
      let yes = block nm ~limits ~depth:(depth + 1) yes in
      let no = block nm ~limits ~depth:(depth + 1) no in
      [ ind ^ "if " ^ p ^ " {" ]
      @ yes
      @
      if no = [] then [ ind ^ "}" ]
      else [ ind ^ "} else {" ] @ no @ [ ind ^ "}" ]
  | Loop_stmt.Charge_scan_update ->
      let limit = i64_lit (Expr.Scan_limits.max_updates limits) in
      line
        (Printf.sprintf "if (scan_remaining <= 0) %s"
           (meter_failure F.Meter.Updates_exhausted limit))
      @ line "scan_remaining -= 1;"
  | Loop_stmt.Mark _ -> []
  | Loop_stmt.Reduce_sum _ ->
      invalid_arg
        "Loop_c: a structured sum is expanded at the entry point \
         (Loop_sum.program)"
  | Loop_stmt.Release_scan_state width ->
      line (Printf.sprintf "scan_live -= %d;" (2 * width))
  | Loop_stmt.Reserve_scan_state width ->
      let live = 2 * width in
      let max_state = Expr.Scan_limits.max_state limits in
      line
        (Printf.sprintf "if (scan_live + %d > %s) %s" live (int_lit max_state)
           (meter_failure F.Meter.State_over_limit (string_of_int max_state)))
      @ line (Printf.sprintf "scan_live += %d;" live)
  | Loop_stmt.Reset_meter ->
      line
        (Printf.sprintf "scan_remaining = %s;"
           (i64_lit (Expr.Scan_limits.max_updates limits)))
      @ line "scan_live = 0;"
  | Loop_stmt.Store { buffer = b; coord = c; value } ->
      line (store nm b (At c) value)
  | Loop_stmt.Store_flat { buffer = b; offset = i; value } ->
      line (store nm b (Flat i) value)

and store nm b addr value =
  let cell () = buffer nm b ^ "[" ^ addr_offset nm b addr ^ "]" in
  match value with
  | Loop_stored.Bool e ->
      (* Evaluation order between the target and the value does not matter:
         both are pure. *)
      let e = num nm e in
      let cell = cell () in
      Printf.sprintf "%s = (%s) != %s ? 1 : 0;" cell e (lit nm 0.)
  | Loop_stored.F32 (Loop_expr.Round_f32 e) | Loop_stored.F32 e ->
      let e = num nm e in
      Printf.sprintf "%s = (float)%s;" (cell ()) e
  | Loop_stored.I64 e ->
      let e = big nm e in
      Printf.sprintf "%s = %s;" (cell ()) e

and block nm ~limits ~depth body = List.concat_map (stmt nm ~limits ~depth) body

and node nm ~limits ~depth (nd : V.node) : string list =
  let ind = indent depth in
  match nd with
  | V.Scalar s -> stmt nm ~limits ~depth s
  | V.Reduction r -> vreduction nm ~depth r
  | V.If (p, yes, no) ->
      let p = pred nm p in
      let yes = nodes nm ~limits ~depth:(depth + 1) yes in
      let no = nodes nm ~limits ~depth:(depth + 1) no in
      [ ind ^ "if " ^ p ^ " {" ]
      @ yes
      @
      if no = [] then [ ind ^ "}" ]
      else [ ind ^ "} else {" ] @ no @ [ ind ^ "}" ]
  | V.Loop { var = v; lo; hi = hi_ix; body } ->
      let name = var nm v in
      let lo = index nm lo in
      let hi = index nm hi_ix in
      let inline, limit =
        match hi_ix with
        | Loop_index.Const _ | Loop_index.Var _ -> ([], hi)
        | _ ->
            let n = bound nm v in
            ([ ind ^ "  const int64_t " ^ n ^ " = " ^ hi ^ ";" ], n)
      in
      [ ind ^ "{" ]
      @ inline
      @ [
          Printf.sprintf "%s  for (int64_t %s = %s; %s < %s; %s++) {" ind name
            lo name limit name;
        ]
      @ nodes nm ~limits ~depth:(depth + 2) body
      @ [ ind ^ "  }"; ind ^ "}" ]
  | V.Vector l -> vloop nm ~depth ~scalar_stmt:(stmt nm ~limits ~depth) l

and nodes nm ~limits ~depth ns = List.concat_map (node nm ~limits ~depth) ns

(* What function scope must declare: the temporaries, in first-assigned order,
   and whether the program touches the scan meter. *)
let declarations (p : Loop_program.t) =
  let floats = ref [] and int64s = ref [] and indices = ref [] in
  let meter = ref false in
  let seen = ref Loop_temp.Set.empty in
  let add r t =
    if not (Loop_temp.Set.mem t !seen) then (
      seen := Loop_temp.Set.add t !seen;
      r := t :: !r)
  in
  let rec go (s : Loop_stmt.t) =
    match s with
    | Loop_stmt.Assign (Loop_carrier.Float, t, _) -> add floats t
    | Loop_stmt.Assign (Loop_carrier.Int64, t, _) -> add int64s t
    | Loop_stmt.Assign_index (t, _) | Loop_stmt.Assign_index_of_i64 (t, _) ->
        add indices t
    | Loop_stmt.For { body; _ } -> List.iter go body
    | Loop_stmt.Reduce_sum _ ->
        invalid_arg
          "Loop_c: a structured sum is expanded at the entry point \
           (Loop_sum.program)"
    | Loop_stmt.If (_, yes, no) ->
        List.iter go yes;
        List.iter go no
    | Loop_stmt.Charge_scan_update | Loop_stmt.Release_scan_state _
    | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter ->
        meter := true
    | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ | Loop_stmt.Fail_if _
    | Loop_stmt.Mark _ | Loop_stmt.Store _ | Loop_stmt.Store_flat _ ->
        ()
  in
  List.iter go p.Loop_program.body;
  (List.rev !floats, List.rev !int64s, List.rev !indices, !meter)

let check_formats (p : Loop_program.t) =
  match
    List.find_opt (fun b -> Option.is_none (cell_type b)) p.Loop_program.buffers
  with
  | None -> Ok ()
  | Some b -> Error (`Unsupported_format (b.Loop_buffer.id, fmt_of b))

let param_type (b : Loop_buffer.t) =
  let t = Option.get (cell_type b) in
  match b.Loop_buffer.role with
  | Loop_buffer.Input -> "const " ^ t
  | Loop_buffer.Output | Loop_buffer.Scratch -> t

let kernel ?vector ?(numerics = Loop_numerics.Reference_f64) ?precision
    ?fuse_reductions ~name (p : Loop_program.t) : (t, [> error ]) Err.t =
  let p = Loop_sum.program p in
  let plan =
    match precision with
    | None -> Ok (Loop_plan.resolve ?target:vector ?fuse_reductions ~numerics p)
    | Some precision ->
        Result.map_error
          (fun r -> `Unsupported_precision r)
          (Loop_plan.force ?target:vector ~precision p)
  in
  match (check_formats p, plan) with
  | Error e, _ -> Err.fail e
  | Ok (), Error e -> Err.fail e
  | Ok (), Ok plan ->
      let precision = plan.Loop_plan.precision in
      let nm =
        {
          vars = Hashtbl.create 8;
          temps = Hashtbl.create 8;
          index_temps = Hashtbl.create 8;
          arrays = Hashtbl.create 8;
          buffers = Hashtbl.create 8;
          f32 = precision = Loop_numerics.Precision.F32;
          sites = F.sites p;
          next_site = 0;
          used = [];
          local_doubles = 0L;
        }
      in
      (* Buffers take their positions in program order, before any use. *)
      let params =
        List.map
          (fun (b : Loop_buffer.t) ->
            Printf.sprintf "%s *%s" (param_type b) (buffer nm b))
          p.Loop_program.buffers
      in
      let limits = p.Loop_program.scan_limits in
      let body =
        match plan.Loop_plan.vector with
        | None -> block nm ~limits ~depth:1 p.Loop_program.body
        | Some vp ->
            (match Err.payload (Loop_vector_check.program vp) with
            | Ok () -> ()
            | Error e ->
                invalid_arg
                  (Fmt.str "Loop_c: the vectorizer built an invalid program: %a"
                     Loop_vector_check.pp_error e));
            nodes nm ~limits ~depth:1 vp.Loop_vector.body
      in
      if nm.next_site <> Array.length nm.sites then
        invalid_arg "Loop_c: a failure site was not written";
      let floats, int64s, indices, meter = declarations p in
      let fty = float_type nm in
      let decl ty init ids name_of =
        List.map
          (fun t -> Printf.sprintf "  %s %s = %s;" ty (name_of nm t) init)
          ids
      in
      (* A temporary the program only assigns would trip the compiler's
         set-but-unused warning: name each once. *)
      let named =
        List.map (fun t -> temp nm t) floats
        @ List.map (fun t -> index_temp nm t) indices
        @ List.map (fun t -> temp nm t) int64s
      in
      let temps =
        decl fty (lit nm 0.) floats temp
        @ decl "int64_t" "0" indices index_temp
        @ decl "int64_t" "0" int64s temp
        @ List.map (fun n -> Printf.sprintf "  (void)%s;" n) named
      in
      let meter =
        if meter then
          [
            Printf.sprintf "  int64_t scan_remaining = %s;"
              (i64_lit (Expr.Scan_limits.max_updates limits));
            "  int64_t scan_live = 0;";
            "  (void)scan_live;";
          ]
        else []
      in
      let params_text =
        String.concat ", "
          ("struct model_error *err" :: "double *local" :: params)
      in
      let voids =
        "  (void)err; (void)local;"
        :: List.map
             (fun (b : Loop_buffer.t) ->
               Printf.sprintf "  (void)%s;" (buffer nm b))
             p.Loop_program.buffers
      in
      let source =
        String.concat "\n"
          ([ Printf.sprintf "static int %s(%s) {" name params_text ]
          @ voids @ temps @ meter @ body @ [ "  return 0;"; "}"; "" ])
      in
      if nm.f32 then use nm R.Name.F32_prelude;
      let used = nm.used in
      Err.return
        {
          source;
          helpers = List.filter (fun n -> List.mem n used) R.Name.all;
          local_doubles = nm.local_doubles;
          precision;
          refusal = plan.Loop_plan.refusal;
          buffer_types =
            List.map (fun b -> Option.get (cell_type b)) p.Loop_program.buffers;
        }

(* The emitter is split: names, scalar expressions and failures in [Loop_c_base],
   the vector loops and scheduled sums in [Loop_c_vector], statements and the
   kernel here. The interface is this module's. *)
type error = Loop_c_base.error

type nonrec t = Loop_c_base.t = {
  source : string;
  helpers : R.Name.t list;
  local_doubles : int64;
  precision : Loop_numerics.Precision.t;
  refusal : Loop_numerics.Refusal.t option;
  buffer_types : string list;
}

let pp_error = Loop_c_base.pp_error
let float_lit = Loop_c_base.float_lit
