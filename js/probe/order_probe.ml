(* Evaluation-order oracle for Stage 0 of the Expr tail-call conversion (see
   the plan under .ai/). Measures, for every order-sensitive [Eval.value] call
   site, which operand a given backend actually evaluates first, using
   logging [Env.load]/[Env.load_index] callbacks driven through the real
   evaluator. Native, bytecode and jsoo build this exact source; each route's
   output is diffed against its own committed golden by
   `make expr_order.runtest`. Do not depend on evaluation order elsewhere in
   this repo -- that is precisely the property this probe exists to pin down
   per backend, not to declare safe. *)

open Expr

(* ---- logging ---- *)

let log : string list ref = ref []
let record tag = log := tag :: !log
let reset_log () = log := []
let trace () = String.concat "," (List.rev !log)

(* ---- fresh sources ---- *)

let next_id = ref 0

let fresh_source () =
  let n = !next_id in
  incr next_id;
  Source.create n

(* ---- registries: dispatch by source identity, log on every call. A source
   is registered into exactly one of these four; [assert false] on a miss
   would mean a case built a [Load]/[Data] node the harness never
   registered. ---- *)

let load_ok : (Source.t * (int Coord.t -> string * float)) list ref = ref []
let load_err : (Source.t * (int Coord.t -> string)) list ref = ref []
let index_ok : (Source.t * (int Coord.t -> string * int64)) list ref = ref []
let index_err : (Source.t * (int Coord.t -> string)) list ref = ref []
let find tbl s = List.find_opt (fun (s', _) -> Source.equal s s') tbl

let env =
  {
    Eval.Env.load =
      (fun s c ->
        match find !load_ok s with
        | Some (_, f) ->
            let tag, v = f c in
            record tag;
            Err.return v
        | None -> (
            match find !load_err s with
            | Some (_, f) ->
                let tag = f c in
                record tag;
                Err.fail (`Unknown_source s)
            | None -> assert false));
    load_index =
      (fun s c ->
        match find !index_ok s with
        | Some (_, f) ->
            let tag, v = f c in
            record tag;
            Err.return v
        | None -> (
            match find !index_err s with
            | Some (_, f) ->
                let tag = f c in
                record tag;
                Err.fail (`Unknown_source s)
            | None -> assert false));
  }

let register_load_ok f =
  let s = fresh_source () in
  load_ok := (s, f) :: !load_ok;
  s

let register_load_err f =
  let s = fresh_source () in
  load_err := (s, f) :: !load_err;
  s

let register_index_ok f =
  let s = fresh_source () in
  index_ok := (s, f) :: !index_ok;
  s

let register_index_err f =
  let s = fresh_source () in
  index_err := (s, f) :: !index_err;
  s

let load_ok_const tag v = register_load_ok (fun _ -> (tag, v))
let load_err_const tag = register_load_err (fun _ -> tag)
let index_ok_const tag v = register_index_ok (fun _ -> (tag, v))
let index_err_const tag = register_index_err (fun _ -> tag)

(* ---- index/value builders ----
   Public smart constructors throughout, per the plan: index constructors are
   private, and [Index.data] produces a Position, while arithmetic needs
   Deltas. *)

(* Coordinate a [Data] node reads AT: irrelevant to this oracle, only the
   [load_index] CALL matters, so every [Data] node below reads at the same
   dummy zero coordinate. *)
let zero_coord = Coord.of_fn (fun _ -> Index.zero)

let data_ok tag ~extent raw : Role.Position.t Index.t =
  Index.data (index_ok_const tag raw) zero_coord extent

let data_err tag : Role.Position.t Index.t =
  Index.data (index_err_const tag) zero_coord 100

let delta_ok tag ~extent raw = Index.of_position (data_ok tag ~extent raw)
let delta_err tag = Index.of_position (data_err tag)
let load_v tag v = Value.load (load_ok_const tag v) zero_coord
let load_v_err tag = Value.load (load_err_const tag) zero_coord
let out_coord = Coord.of_fn (fun _ -> 0)

(* ---- case runner ---- *)

let eval ?scan ?scan_meter e =
  Eval.value ?scan ?scan_meter env ~output:out_coord e

let show_result = function
  | Ok v -> Printf.sprintf "ok %g" v
  | Error e ->
      Printf.sprintf "error %s"
        (Format.asprintf "%a" Eval.pp_error (Err.Error.kind e))

let case name e =
  reset_log ();
  let r = eval e in
  Printf.printf "%s\n  result: %s\n  trace: %s\n" name (show_result r)
    (trace ())

let case_scan name ~scan_meter e =
  reset_log ();
  let r = eval ~scan_meter e in
  Printf.printf "%s\n  result: %s\n  trace: %s\n" name (show_result r)
    (trace ())

let case_local_scan name ~scan e =
  reset_log ();
  let r = eval ~scan e in
  Printf.printf "%s\n  result: %s\n  trace: %s\n" name (show_result r)
    (trace ())

(* ==== Seven rewritten order-sensitive call sites (14 cases) ==== *)

let select_probe c = Value.select c (Value.const 1.0) (Value.const 0.0)

let reduce_probe ~lo ~hi =
  Builder.run
    (Builder.reduction ~kind:Reduction.Sum ~lo ~hi (fun _ ->
         Builder.return (Value.const 0.)))

let scan_limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:100 ~max_updates:1000L)

let trivial_scan =
  Err.or_raise ~pp_error:Scan.pp_error
    (Builder.run
       (Builder.scan ~limits:scan_limits ~width:1 ~steps:0
          ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
          ~update:(fun ~step:_ ~lane:_ ~previous_at:_ ->
            Builder.return (Value.const 0.))))

let local_scan_id = Builder.run Builder.fresh_local

let always_ok_scan_reader : Eval.scan_reader =
 fun _ ~row:_ ~lane:_ -> Err.return 9.0

let make_out ~h ~w = Coord.set (Coord.set zero_coord Axis.H h) Axis.W w

let max_pool_descriptor ~h ~w ~result =
  Err.or_raise ~pp_error:Intrinsic.pp_error
    (Intrinsic.max_pool
       ~source:(load_ok_const "pixel" 7.0)
       ~in_h:1 ~in_w:1 ~kernel_h:1 ~kernel_w:1 ~stride_h:1 ~stride_w:1 ~pad_h:0
       ~pad_w:0 ~out:(make_out ~h ~w) ~result)

let run_order_sensitive_sites () =
  case "Binary/ok" (Value.add (load_v "a" 1.0) (load_v "b" 2.0));
  case "Binary/fail" (Value.add (load_v_err "a") (load_v_err "b"));

  case "Value_lt/ok"
    (select_probe (Bool.value_lt (load_v "a" 1.0) (load_v "b" 2.0)));
  case "Value_lt/fail"
    (select_probe (Bool.value_lt (load_v_err "a") (load_v_err "b")));

  case "Index_eq/ok"
    (select_probe
       (Bool.index_eq
          (delta_ok "a" ~extent:100 0L)
          (delta_ok "b" ~extent:100 0L)));
  case "Index_eq/fail"
    (select_probe (Bool.index_eq (delta_err "a") (delta_err "b")));

  case "Reduce/ok"
    (reduce_probe
       ~lo:(data_ok "lo" ~extent:100 0L)
       ~hi:(delta_ok "hi" ~extent:100 1L));
  case "Reduce/fail" (reduce_probe ~lo:(data_err "lo") ~hi:(delta_err "hi"));

  (let meter = Scan_meter.create ~limits:scan_limits in
   case_scan "Scan_at/ok" ~scan_meter:meter
     (Value.scan_at trivial_scan
        ~row:(data_ok "row" ~extent:100 0L)
        ~lane:(data_ok "lane" ~extent:100 0L)));
  (let meter = Scan_meter.create ~limits:scan_limits in
   case_scan "Scan_at/fail" ~scan_meter:meter
     (Value.scan_at trivial_scan ~row:(data_err "row") ~lane:(data_err "lane")));

  case_local_scan "Local_scan_at/ok" ~scan:always_ok_scan_reader
    (Value.local_scan_at local_scan_id
       ~row:(data_ok "row" ~extent:100 0L)
       ~lane:(data_ok "lane" ~extent:100 0L));
  case_local_scan "Local_scan_at/fail" ~scan:always_ok_scan_reader
    (Value.local_scan_at local_scan_id ~row:(data_err "row")
       ~lane:(data_err "lane"));

  case "Intrinsic.window/ok"
    (Value.intrinsic
       (max_pool_descriptor
          ~h:(data_ok "h" ~extent:100 0L)
          ~w:(data_ok "w" ~extent:100 0L)
          ~result:Intrinsic.Max_pool.Value));
  case "Intrinsic.window/fail"
    (Value.intrinsic
       (max_pool_descriptor ~h:(data_err "h") ~w:(data_err "w")
          ~result:Intrinsic.Max_pool.Value))

(* ==== Helper-preservation cases: Load, Max_pool, Index.Add/Max/Min, nested
   Index.Data (12 cases) ==== *)

let run_helper_cases () =
  (let coord =
     Coord.make
       ~n:(data_ok "N" ~extent:100 10L)
       ~t:(data_ok "T" ~extent:100 11L)
       ~d:(data_ok "D" ~extent:100 12L)
       ~h:(data_ok "H" ~extent:100 13L)
       ~w:(data_ok "W" ~extent:100 14L)
       ~c:(data_ok "C" ~extent:100 15L)
   in
   let value_src =
     register_load_ok (fun c ->
         ( Printf.sprintf "load(n=%d,t=%d,d=%d,h=%d,w=%d,c=%d)"
             (Coord.get c Axis.N) (Coord.get c Axis.T) (Coord.get c Axis.D)
             (Coord.get c Axis.H) (Coord.get c Axis.W) (Coord.get c Axis.C),
           42.0 ))
   in
   case "Load/ok" (Value.load value_src coord));
  (let coord =
     Coord.make ~n:(data_err "N") ~t:(data_err "T") ~d:(data_err "D")
       ~h:(data_err "H") ~w:(data_err "W") ~c:(data_err "C")
   in
   let value_src = load_ok_const "load" 42.0 in
   case "Load/fail" (Value.load value_src coord));

  (let out =
     Coord.make
       ~n:(data_ok "N" ~extent:100 0L)
       ~t:(data_ok "T" ~extent:100 0L)
       ~d:(data_ok "D" ~extent:100 0L)
       ~h:Index.zero ~w:Index.zero
       ~c:(data_ok "C" ~extent:100 0L)
   in
   let d =
     Err.or_raise ~pp_error:Intrinsic.pp_error
       (Intrinsic.max_pool
          ~source:(load_ok_const "pixel" 7.0)
          ~in_h:1 ~in_w:1 ~kernel_h:1 ~kernel_w:1 ~stride_h:1 ~stride_w:1
          ~pad_h:0 ~pad_w:0 ~out ~result:Intrinsic.Max_pool.Value)
   in
   case "Max_pool/ok" (Value.intrinsic d));
  (let out =
     Coord.make ~n:(data_err "N") ~t:(data_err "T") ~d:(data_err "D")
       ~h:Index.zero ~w:Index.zero ~c:(data_err "C")
   in
   let d =
     Err.or_raise ~pp_error:Intrinsic.pp_error
       (Intrinsic.max_pool
          ~source:(load_ok_const "pixel" 7.0)
          ~in_h:1 ~in_w:1 ~kernel_h:1 ~kernel_w:1 ~stride_h:1 ~stride_w:1
          ~pad_h:0 ~pad_w:0 ~out ~result:Intrinsic.Max_pool.Value)
   in
   case "Max_pool/fail" (Value.intrinsic d));

  case "Index.Add/ok"
    (Value.value_of_index
       (Index.add (delta_ok "a" ~extent:100 3L) (delta_ok "b" ~extent:100 4L)));
  case "Index.Add/fail"
    (Value.value_of_index (Index.add (delta_err "a") (delta_err "b")));

  case "Index.Max/ok"
    (Value.value_of_index
       (Index.max (delta_ok "a" ~extent:100 3L) (delta_ok "b" ~extent:100 4L)));
  case "Index.Max/fail"
    (Value.value_of_index (Index.max (delta_err "a") (delta_err "b")));

  case "Index.Min/ok"
    (Value.value_of_index
       (Index.min (delta_ok "a" ~extent:100 3L) (delta_ok "b" ~extent:100 4L)));
  case "Index.Min/fail"
    (Value.value_of_index (Index.min (delta_err "a") (delta_err "b")));

  (let coord =
     Coord.make
       ~n:(data_ok "in_N" ~extent:100 1L)
       ~t:(data_ok "in_T" ~extent:100 2L)
       ~d:(data_ok "in_D" ~extent:100 3L)
       ~h:(data_ok "in_H" ~extent:100 4L)
       ~w:(data_ok "in_W" ~extent:100 5L)
       ~c:(data_ok "in_C" ~extent:100 6L)
   in
   let outer_src = index_ok_const "outer" 0L in
   case "Nested_data/ok"
     (Value.value_of_index (Index.of_position (Index.data outer_src coord 100))));
  let coord =
    Coord.make ~n:(data_err "in_N") ~t:(data_err "in_T") ~d:(data_err "in_D")
      ~h:(data_err "in_H") ~w:(data_err "in_W") ~c:(data_err "in_C")
  in
  let outer_src = index_ok_const "outer" 0L in
  case "Nested_data/fail"
    (Value.value_of_index (Index.of_position (Index.data outer_src coord 100)))

let run () =
  run_order_sensitive_sites ();
  run_helper_cases ()

let () = run ()
