(* Section 4: the engine itself. Not melange-reachable.

   Three things, in increasing order of how much machinery they drag in:

   1. The 16-bit software codecs. Their inputs are chosen for sign and high bit
      patterns, because that is where the int-width fix in lib/native/half.ml
      bites -- linking Payload compiles those branches, only calling them proves
      anything.
   2. A real conv2d+relu built with Graph_builder and run through Eval_direct:
      Bigarray reads and writes, float32 arithmetic, and the printed tensor.
   3. The op walk, whose oracle is self-contained (Direct vs Symbolic must agree
      at every step) and whose configs come from the seeded PCG. *)

let half_inputs =
  [ 1.0; -1.0; 0.1; -0.1; 65504.0; -65504.0; 1e-8; -3.14159; 0.0; -0.0 ]

let run_half () =
  print_endline "=== half ===";
  List.iter
    (fun x ->
      let f = Walk_core.Float32.to_f32 x in
      let bf = Half.Bf16.of_float f and h = Half.Half.of_float f in
      Printf.printf "%s bf16=0x%04x->%s half=0x%04x->%s\n"
        (Walk_core.Float32.to_hex f)
        bf
        (Walk_core.Float32.to_hex (Half.Bf16.to_float bf))
        h
        (Walk_core.Float32.to_hex (Half.Half.to_float h)))
    half_inputs

(* A 4x4x2 input, a 3-filter 2x2 kernel, relu on the output. Small enough that
   the whole result tensor prints, big enough to exercise the windowed loops. *)
let run_conv () =
  print_endline "=== conv2d-eval ===";
  let s ~n ~h ~w ~c = Vec6.shape ~n ~t:1 ~d:1 ~h ~w ~c in
  let axis : Conv.Conv2d.axis_window =
    {
      kernel = Dim.extent 2;
      stride = Op_config.Pos.of_int 1;
      pad_before = Op_config.Nonneg.of_int 0;
      pad_after = Op_config.Nonneg.of_int 0;
      dilation = Op_config.Pos.of_int 1;
    }
  in
  let params : Conv.Conv2d.params =
    {
      h = axis;
      w = axis;
      in_channels = Dim.extent 2;
      groups = Op_config.Pos.of_int 1;
    }
  in
  let x_shape = s ~n:1 ~h:4 ~w:4 ~c:2 in
  let w_shape = s ~n:3 ~h:2 ~w:2 ~c:2 in
  let built =
    Graph_builder.build ~name:"conv"
      ~outputs:(fun y -> [ y ])
      (let open Graph_builder in
       let* x = input ~shape:x_shape () in
       let* weight = constant ~shape:w_shape () in
       let* y = conv2d params ~x ~weight () in
       relu y)
  in
  (* Every failure below RAISES rather than printing. This is a differential
     harness: both sides run this same source, so a fault that reproduces on both
     -- which a broken builder or evaluator would -- prints identical text, diffs
     clean, and passes. Only a non-zero exit is noticed. Same rule as
     [[testing_strategy]]'s "fixture failures abort; they do not become
     absence"; [Err.or_raise ~pp_error:] for a [Err.t], [failwith] for the plain
     [result] Jsont hands back. *)
  let graph = Err.or_raise ~pp_error:Graph_builder.pp_error built in
  (* Deterministic ramps rather than PCG draws: this section is about the
     evaluator, and a fixed pattern keeps it readable in the golden.

     Every value is k/2^n, so it is exact in both float64 and float32 and no
     rounding happens on the way in. That is not fussiness. ocamlopt contracts
     `a *. b -. c` into a fused multiply-add; the bytecode interpreter -- and
     therefore js_of_ocaml -- does not. An earlier ramp here was
     `float_of_int (i + 1) *. scale -. 0.7`, whose 14th element is 0.7 -. 0.7
     computed two ways: ocamlopt returned 0x1.8p-54 and node 0x1p-53, and this
     probe reported an engine divergence that was really a compiler one.
     Products of f32 values are exact in float64 (24+24 < 53 bits), so FMA
     contraction cannot change what the engine computes -- but it can change how
     a probe manufactures its inputs. *)
  let ramp shape ~period ~centre ~denom =
    Tensor.materialize shape (fun c ->
        let i = (Vec6.offset shape c :> int) in
        float_of_int ((i mod period) - centre) /. denom)
  in
  let x_id, w_id =
    match graph.Graph_ir.Graph.inputs with
    | [ x; w ] -> (x, w)
    | ids ->
        failwith
          (Printf.sprintf "probe: expected 2 graph inputs, got %d"
             (List.length ids))
  in
  let env =
    Eval_direct.run graph
      ~constants:[ (w_id, ramp w_shape ~period:13 ~centre:6 ~denom:32.0) ]
      ~inputs:[ (x_id, ramp x_shape ~period:23 ~centre:11 ~denom:8.0) ]
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  List.iter
    (fun oid ->
      let t = Tensor_id.Map.find oid env in
      Format.printf "out %a@." Tensor.pp t;
      match Graph_json.encode_tensor t with
      | Ok j -> Printf.printf "out-json %s\n" j
      | Error e -> failwith ("probe: encode_tensor: " ^ e))
    graph.Graph_ir.Graph.outputs

(* A Direct-vs-Symbolic mismatch prints its verdict and stops that op's walk --
   identically on every backend, so the diff stays clean. The walk's own boolean
   is the only thing that distinguishes "agreed everywhere" from "disagreed and
   gave up", and it has to reach the exit status. Every op runs before failing,
   so one bad op does not hide the rest. *)
let run_walk () =
  print_endline "=== op-walk ===";
  let ppf = Format.std_formatter in
  let failures =
    List.concat
      (List.mapi
         (fun i m ->
           let ok =
             Native_op_walk.run m ~ppf
               ~pcg:(Walk_core.Pcg.seed ~seed:(Int64.of_int i) ~seq:1L)
               ~steps:3
           in
           if ok then [] else [ i ])
         Native_op_walk.all_walks)
  in
  Format.pp_print_flush ppf ();
  if failures <> [] then
    failwith
      (Printf.sprintf "probe: %d op walk(s) failed to verify (indices %s)"
         (List.length failures)
         (String.concat "," (List.map string_of_int failures)))

let run () =
  run_half ();
  run_conv ();
  run_walk ()
