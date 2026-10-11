(* Run the real ATen / Native pair on one model and sample through the shared
   frontier coordinator and print what it finds. *)
open Inference_compare_schema
module P = Inference_compare_pt2
module Plan = Inference_compare_plan
module Core = Inference_compare_core
module Ex = Inference_compare_extract
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm

let with_batch (tensor : Pt2_tensor.t) =
  let stride = Pt2_tensor.numel tensor in
  {
    tensor with
    Pt2_tensor.sizes = 1 :: tensor.sizes;
    strides = stride :: tensor.strides;
  }

let donor : Side.t =
  match Sys.getenv_opt "DONOR" with Some "right" -> Right | _ -> Left

let calibrate = Sys.getenv_opt "CALIBRATE" <> None

let tol =
  match Sys.getenv_opt "TOL" with Some s -> float_of_string s | None -> 1e-4

let () =
  let model = Sys.argv.(1) and inputs_pt = Sys.argv.(2) in
  let sample =
    match Pt2_archive.load_pt_tensor_map inputs_pt with
    | Ok ((_, t) :: _) -> with_batch t
    | _ -> failwith "inputs"
  in
  let aten0 =
    match P.Aten_model.of_file ~hashers ~algorithm model with
    | Ok a -> a
    | Error m -> failwith m
  in
  let input_name =
    match
      List.find_opt
        (fun (s : P.Aten_model.source) -> s.captured = None)
        aten0.sources
    with
    | Some s -> s.name
    | None -> failwith "no user input"
  in
  let four = Sys.getenv_opt "RIGHT" = Some "native4d" in
  let pair4d = ref None in
  let mapping, left_graph, right_graph, initial, node_attributes, right_endpoint
      =
    if four then (
      match
        P.Pair4d.load ~hashers ~algorithm ~path:model
          ~inputs:[ (input_name, sample) ]
      with
      | Error m -> failwith ("pair4d: " ^ m)
      | Ok p ->
          pair4d := Some p;
          ( P.Pair4d.mapping p,
            P.Pair4d.left_graph p,
            P.Pair4d.right_graph p,
            P.Pair4d.initial p,
            P.Pair4d.node_attributes p,
            P.Pair4d.native4d_endpoint p ))
    else
      match
        P.Pair.load ~hashers ~algorithm ~path:model
          ~inputs:[ (input_name, sample) ]
      with
      | Error m -> failwith ("pair: " ^ m)
      | Ok p ->
          ( P.Pair.mapping p,
            P.Pair.left_graph p,
            P.Pair.right_graph p,
            P.Pair.initial p,
            P.Pair.node_attributes p,
            P.Pair.native_endpoint p )
  in
  let left_endpoint =
    Core.Side_exec.Endpoint
      {
        runner = P.Aten_runner.create ();
        build =
          (fun ex ->
            Ex.Dialect.realize (module P.Aten_dialect.Dialect) aten0 ex);
      }
  in
  if Sys.getenv_opt "DEBUG_INIT" <> None then begin
    let have =
      List.map (fun (k, _) -> Id.Canonical_value.to_string k) initial
    in
    List.iter
      (fun (i : Initial_binding.t) ->
        let k = Id.Canonical_value.to_string i.key in
        if not (List.mem k have) then Printf.printf "no value for %s\n" k)
      mapping.frontier.initial_values
  end;
  (match (!pair4d, Sys.getenv_opt "ASSOC") with
  | Some p, Some _ ->
      (* the composition regrouped: ATen.(canonicalize.lower) against the
         (ATen.canonicalize).lower the pair itself uses *)
      let get = function Ok x -> x | Error m -> failwith m in
      let c = get (Lazy.force p.four.canonicalize)
      and l = get (Lazy.force p.four.lower) in
      let compose a b =
        match
          Inference_compare_compose.Compose.compose ~hashers ~algorithm a b
        with
        | Ok m -> m
        | Error e ->
            failwith
              (Format.asprintf "compose: %a"
                 (Err.Error.pp_kind Inference_compare_compose.Compose_error.pp)
                 e)
      in
      let alt = compose p.base.import.mapping (compose c.mapping l.mapping) in
      Inference_compare_test.Compose_cases.same_meaning
        "(ATen.canonicalize).lower vs ATen.(canonicalize.lower)" mapping alt
  | _ -> ());
  let r, run_ctx =
    Pair_harness.Harness.run ~donor ~tol ~mapping ~left_graph ~right_graph
      ~initial ~node_attributes ~left_endpoint ~right_endpoint
  in
  if calibrate then begin
    (* The donor's segmented run (each unit fed the donor's own committed
       values) against that engine's whole-graph run, bitwise. *)
    let inputs = [ (input_name, sample) ] in
    let archive = aten0.archive in
    let whole : (string list, string) result =
      match donor with
      | Left -> (
          match Interp.run_named archive ~inputs with
          | Error _ -> Error "whole ATen run failed"
          | Ok ts ->
              Ok
                (List.map2
                   (fun t (_, v) ->
                     match P.Conv.logical_of_aten run_ctx (L.spec v) t with
                     | Ok w -> L.contents w
                     | Error m -> m)
                   ts r.finals))
      | Right when four -> (
          let p = Option.get !pair4d in
          let g = p.four.n4.graph in
          let user =
            List.find
              (fun i -> Graph_common.input_kind g i = Graph_common.Input.Input)
              g.inputs
          in
          let v = List.assoc input_name p.base.case.values in
          let six =
            match P.Graph_model.spec_of g user with
            | Error m -> failwith m
            | Ok sp -> (
                match L.make run_ctx sp (L.contents v) with
                | Ok x -> x
                | Error _ -> failwith "six-axis input")
          in
          match P.Conv.native_of_logical six with
          | Error m -> Error m
          | Ok packed -> (
              match
                Native4d.Eval_direct4.run g
                  ~constants:
                    (Graph_ir.Tensor_id.Map.bindings
                       (Result.get_ok (Lazy.force p.four.constants)))
                  ~inputs:[ (user, packed) ]
              with
              | Error _ -> Error "whole Native4D run failed"
              | Ok env ->
                  Ok
                    (List.map2
                       (fun id (_, v) ->
                         let (Tensor.Tensor t as pk) =
                           Graph_ir.Tensor_id.Map.find id env
                         in
                         let shape =
                           List.map
                             (fun a -> Int64.of_int (Vec6.get t.shape a :> int))
                             Axis.all
                         in
                         let spec =
                           { (L.spec v) with Tensor_spec.logical_shape = shape }
                         in
                         match P.Conv.logical_of_native run_ctx spec pk with
                         | Ok w -> L.contents w
                         | Error m -> m)
                       g.outputs r.finals)))
      | Right -> (
          match Native_interp.run_named archive ~inputs with
          | Error _ -> Error "whole Native run failed"
          | Ok ts ->
              Ok
                (List.map2
                   (fun (Tensor.Tensor t as p) (_, v) ->
                     let shape =
                       List.map
                         (fun a -> Int64.of_int (Vec6.get t.shape a :> int))
                         Axis.all
                     in
                     let spec =
                       { (L.spec v) with Tensor_spec.logical_shape = shape }
                     in
                     match P.Conv.logical_of_native run_ctx spec p with
                     | Ok w -> L.contents w
                     | Error m -> m)
                   ts r.finals))
    in
    match whole with
    | Error m -> Printf.printf "calibration: %s\n" m
    | Ok whole ->
        let same =
          List.for_all2
            (fun w (_, v) -> String.equal w (L.contents v))
            whole r.finals
        in
        if not same then
          List.iter2
            (fun w (_, v) ->
              let n = String.length w / 4 in
              let worst = ref 0.0 and diff = ref 0 in
              for i = 0 to n - 1 do
                let a = Int32.float_of_bits (String.get_int32_le w (4 * i))
                and b =
                  Int32.float_of_bits
                    (String.get_int32_le (L.contents v) (4 * i))
                in
                if a <> b then incr diff;
                worst := Float.max !worst (Float.abs (a -. b))
              done;
              Printf.printf "  %d of %d elements differ, max abs %.3g\n" !diff n
                !worst)
            whole r.finals;
        Printf.printf "calibration: %d final output(s) %s the whole-graph run\n"
          (List.length whole)
          (if same then "bitwise equal" else "DIFFER from")
  end

let () = if Sys.getenv_opt "DEBUG_INIT" <> None then ()
