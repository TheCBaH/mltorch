(* Run a decoded verification spec through the ATen-vs-native harness.

   [to_node] synthesizes each tensor input (one PCG instance, seeded by the
   spec's [seed], threaded through every draw so the case is reproducible) and
   lowers the arguments to a single-node Pytorch_types.Node — exactly the shape
   the interpreter and the verifier already consume. [run] then executes that
   node on both paths (Interp_dispatch for ATen, Op_bridge for native) and
   compares the outputs with Verify, returning whether they agreed.

   The number of output tensors to allocate for the comparison comes from the
   generated Aten_op_config (the same op's return arity). *)

open Pytorch_types
open Schema_runtime
module Tspec = Aten_spec.Tensor_spec
module Sv = Aten_spec.Scalar_value
module Av = Aten_spec.Arg_value
module Pcg = Aten_spec.Pcg

let numel shape = List.fold_left ( * ) 1 shape

let scalar_to_float = function
  | Sv.Int i -> Aten_spec.Float32.to_f32 (float_of_int i)
  | Sv.Float f -> f
  | Sv.Bool b -> if b then 1.0 else 0.0

(* Sample one f32 from a distribution, advancing the RNG. *)
let sample_f32 (d : Aten_spec.Distribution.t) pcg =
  match d with
  | Uniform { low; high } -> Pcg.uniform ~low ~high pcg
  | Normal { mean; variance } -> Pcg.normal ~mean ~std:(sqrt variance) pcg

(* Fill a float32 Bigarray from a payload source, threading [pcg]. *)
let fill_f32 shape (source : Tspec.source) pcg =
  let n = numel shape in
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let pcg =
    match source with
    | Tspec.Values vs ->
        let arr = Array.of_list vs in
        if Array.length arr <> n then
          failwith
            (Printf.sprintf "spec: values has %d entries, shape needs %d"
               (Array.length arr) n);
        let rec go i =
          if i < n then (
            ba.{i} <- scalar_to_float arr.(i);
            go (i + 1))
        in
        go 0;
        pcg
    | Tspec.Sequence { start; step } ->
        let rec go i =
          if i < n then (
            ba.{i} <-
              Aten_spec.Float32.to_f32 (start +. (float_of_int i *. step));
            go (i + 1))
        in
        go 0;
        pcg
    | Tspec.Random dist ->
        let rec go i pcg =
          if i >= n then pcg
          else
            let v, pcg = sample_f32 dist pcg in
            ba.{i} <- v;
            go (i + 1) pcg
        in
        go 0 pcg
  in
  (pcg, ba)

(* Fill an int64 Bigarray (exact match path; no float distributions). *)
let fill_i64 shape (source : Tspec.source) pcg =
  let n = numel shape in
  let ba = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout n in
  (match source with
  | Tspec.Values vs ->
      let arr = Array.of_list vs in
      if Array.length arr <> n then
        failwith
          (Printf.sprintf "spec: values has %d entries, shape needs %d"
             (Array.length arr) n);
      let rec go i =
        if i < n then (
          ba.{i} <- Int64.of_float (scalar_to_float arr.(i));
          go (i + 1))
      in
      go 0
  | Tspec.Sequence { start; step } ->
      let rec go i =
        if i < n then (
          ba.{i} <- Int64.of_float (start +. (float_of_int i *. step));
          go (i + 1))
      in
      go 0
  | Tspec.Random _ ->
      failwith "spec: random payloads are only supported for f32 tensors");
  (pcg, ba)

let synthesize pcg (ts : Tspec.t) =
  match ts.dtype with
  | Aten_spec.Dtype.F32 ->
      let pcg, ba = fill_f32 ts.shape ts.source pcg in
      (pcg, Aten_tensor.of_bigarray Aten_dtype.float32 ba ts.shape)
  | Aten_spec.Dtype.I64 ->
      let pcg, ba = fill_i64 ts.shape ts.source pcg in
      (pcg, Aten_tensor.of_bigarray Aten_dtype.int64 ba ts.shape)
  | _ -> failwith "spec: tensor synthesis supports f32 and i64 only"

let scalar_to_arg = function
  | Sv.Int i -> Argument.Int i
  | Sv.Float f -> Argument.Float f
  | Sv.Bool b -> Argument.Bool b

(* Lower one argument: synthesize tensors (binding them in [env] under [name])
   and translate everything to a Pytorch_types.Argument. *)
let lower pcg env name (av : Av.t) =
  let tensor pcg env nm spec =
    let pcg, t = synthesize pcg spec in
    (pcg, String_map.add nm t env, TensorArgument.make nm)
  in
  match av with
  | Av.Tensor spec ->
      let pcg, env, ta = tensor pcg env name spec in
      (pcg, env, Argument.Tensor ta)
  | Av.Tensor_opt None -> (pcg, env, Argument.None false)
  | Av.Tensor_opt (Some spec) ->
      let pcg, env, ta = tensor pcg env name spec in
      (pcg, env, Argument.Tensor ta)
  | Av.Tensor_list specs ->
      let pcg, env, tas, _ =
        List.fold_left
          (fun (pcg, env, acc, k) spec ->
            let nm = Printf.sprintf "%s_%d" name k in
            let pcg, env, ta = tensor pcg env nm spec in
            (pcg, env, ta :: acc, k + 1))
          (pcg, env, [], 0) specs
      in
      (pcg, env, Argument.Tensors (List.rev tas))
  | Av.Int i -> (pcg, env, Argument.Int i)
  | Av.Int_opt None -> (pcg, env, Argument.None false)
  | Av.Int_opt (Some i) -> (pcg, env, Argument.Int i)
  | Av.Int_list xs -> (pcg, env, Argument.Ints xs)
  | Av.Int_list_opt None -> (pcg, env, Argument.None false)
  | Av.Int_list_opt (Some xs) -> (pcg, env, Argument.Ints xs)
  | Av.Float f -> (pcg, env, Argument.Float f)
  | Av.Float_opt None -> (pcg, env, Argument.None false)
  | Av.Float_opt (Some f) -> (pcg, env, Argument.Float f)
  | Av.Bool b -> (pcg, env, Argument.Bool b)
  | Av.Bool_opt None -> (pcg, env, Argument.None false)
  | Av.Bool_opt (Some b) -> (pcg, env, Argument.Bool b)
  | Av.Scalar sv -> (pcg, env, scalar_to_arg sv)
  | Av.Scalar_opt None -> (pcg, env, Argument.None false)
  | Av.Scalar_opt (Some sv) -> (pcg, env, scalar_to_arg sv)
  | Av.Str s -> (pcg, env, Argument.String s)

let outputs_for target =
  let arity =
    match Aten_op_config.find target with
    | Some c -> (
        match c.returns with
        | Aten_op_config.Single -> 1
        | Aten_op_config.Tuple2 -> 2
        | Aten_op_config.Tuple3 -> 3)
    | None -> 1
  in
  List.init arity (fun i ->
      Argument.Tensor (TensorArgument.make (Printf.sprintf "out%d" i)))

let to_node ?(pcg0 = Pcg.default) (spec : Aten_spec.Op_spec.t) =
  let _pcg, env, inputs_rev =
    List.fold_left
      (fun (pcg, env, acc) (name, av) ->
        let pcg, env, arg = lower pcg env name av in
        (pcg, env, NamedArgument.make name arg None :: acc))
      (pcg0, String_map.empty, [])
      spec.args
  in
  let node =
    Node.make spec.target (List.rev inputs_rev) (outputs_for spec.target)
      String_map.empty None (Some "spec")
  in
  (env, node)

let pp_interp_error ppf = function
  | #Interp_decode.error as e -> Interp_decode.pp_error ppf e
  | `Aten_runtime_failure (op, st) ->
      Fmt.pf ppf "ATen op %s failed with status %d" op st
  | `Unhandled_op target -> Fmt.pf ppf "unhandled op %S" target

(* Execute the spec on both paths and report. Returns [true] if the native and
   ATen outputs agreed (or the op has no native impl, which is reported and
   treated as a pass), [false] on mismatch or error. *)
let run ?(ppf = Format.std_formatter) (spec : Aten_spec.Op_spec.t) : bool =
  let env, node = to_node spec in
  match Interp_dispatch.dispatch env node with
  | Error e -> (
      match Err.Error.kind e with
      | `Unhandled_op _ ->
          Format.fprintf ppf "[spec] %s: skipped (aten interp: unhandled op)@."
            node.target;
          true
      | kind ->
          Format.fprintf ppf "[spec] %s: aten interp error: %a@." node.target
            pp_interp_error kind;
          false)
  | Ok env' -> (
      match Op_bridge.dispatch ~aten_env:env node with
      | None ->
          Format.fprintf ppf "[spec] %s: skipped (no native impl)@." node.target;
          true
      | Some (Error e) ->
          Format.fprintf ppf "[spec] %s: bridge error: %a@." node.target
            Op_bridge.pp_error (Err.Error.kind e);
          false
      | Some (Ok (graph, bindings)) -> (
          match Eval_direct.run graph ~inputs:bindings with
          | Error e ->
              Format.fprintf ppf "[spec] %s: eval error: %a@." node.target
                Eval_direct.pp_error (Err.Error.kind e);
              false
          | Ok result_env ->
              let native_outputs =
                List.map
                  (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
                  graph.Graph_ir.Graph.outputs
              in
              let atol = Verify.atol_for_target node.target in
              let errors =
                Verify.verify_node ~atol ~aten_env:env' node native_outputs
              in
              if errors = [] then (
                Format.fprintf ppf "[spec] %s: matched@." node.target;
                true)
              else (
                Verify.report ppf node.target errors;
                false)))

let pp_aten ppf t =
  match Aten_tensor.as_float32 t with
  | Some ba -> Aten_tensor.pp_float32 ppf ba
  | None -> (
      match Aten_tensor.as_int64 t with
      | Some ba -> Aten_tensor.pp_int64 ppf ba
      | None -> Format.pp_print_string ppf "<unsupported dtype>")

(* A native output is a dense 6D tensor; flatten it to a 1D ATen tensor (same
   element order, see native_aten_bridge_design) so it prints with the same
   formatter as the ATen output. *)
let pp_native ppf packed =
  Core.Pretty.result ~ok:pp_aten
    ~error:(fun ppf e -> Fmt.pf ppf "<error: %s>" e)
    ppf
    (Tensor_bridge.to_aten_flat packed)

(* Evaluate the spec on both paths and print each output tensor, ATen above
   native, so the two can be compared by eye. *)
let eval_print ?(ppf = Format.std_formatter) (spec : Aten_spec.Op_spec.t) : unit
    =
  let env, node = to_node spec in
  match Interp_dispatch.dispatch env node with
  | Error e ->
      Format.fprintf ppf "[eval] %s@.  aten = <error: %a>@." node.target
        pp_interp_error (Err.Error.kind e)
  | Ok env' ->
      let out_names =
        List.filter_map
          (function
            | Argument.Tensor ta -> Some ta.TensorArgument.name | _ -> None)
          node.outputs
      in
      let native =
        match Op_bridge.dispatch ~aten_env:env node with
        | None -> `None
        | Some (Error e) -> `Bridge_error e
        | Some (Ok (graph, bindings)) -> (
            match Eval_direct.run graph ~inputs:bindings with
            | Error e -> `Eval_error e
            | Ok result_env ->
                `Ok
                  (List.map
                     (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
                     graph.Graph_ir.Graph.outputs))
      in
      Format.fprintf ppf "[eval] %s@." node.target;
      List.iteri
        (fun i name ->
          Format.fprintf ppf "  aten   %s = %a@." name
            (Core.Pretty.option_or ~none:"<missing>" pp_aten)
            (String_map.find_opt name env');
          match native with
          | `None ->
              if i = 0 then Format.fprintf ppf "  native = <no native impl>@."
          | `Bridge_error e ->
              if i = 0 then
                Format.fprintf ppf "  native = <error: %a>@." Op_bridge.pp_error
                  (Err.Error.kind e)
          | `Eval_error e ->
              if i = 0 then
                Format.fprintf ppf "  native = <error: eval error: %a>@."
                  Eval_direct.pp_error (Err.Error.kind e)
          | `Ok outs ->
              Format.fprintf ppf "  native %s = %a@." name
                (Core.Pretty.option_or ~none:"<missing>" pp_native)
                (List.nth_opt outs i))
        out_names

(* Deliberately not [Fmt.brackets], which boxes its content and so may
   line-wrap; the promoted cram goldens this feeds assume a single line
   regardless of width, matching the original [Format.fprintf "[%a]" ...]. *)
let pp_shape ppf shape =
  Fmt.pf ppf "[%a]" (Fmt.list ~sep:(Fmt.any ",") Fmt.int) shape

(* The payload/distribution a tensor arg's contents are synthesized from —
   shown alongside its dtype/shape so a reader can see that a walk step's
   "configuration" (shape + this) really is unchanged from step to step; only
   the concrete draw from it differs. *)
let pp_source ppf (s : Tspec.source) =
  match s with
  | Tspec.Values _ -> Fmt.string ppf "values"
  | Tspec.Sequence { start; step } ->
      Fmt.pf ppf "sequence(start=%g,step=%g)" start step
  | Tspec.Random (Uniform { low; high }) ->
      Fmt.pf ppf "uniform(low=%g,high=%g)" low high
  | Tspec.Random (Normal { mean; variance }) ->
      Fmt.pf ppf "normal(mean=%g,variance=%g)" mean variance

let pp_tensor_spec ppf (ts : Tspec.t) =
  Fmt.pf ppf "%s%a~%a"
    (Aten_spec.Dtype.to_string ts.dtype)
    pp_shape ts.shape pp_source ts.source

let pp_scalar_value ppf = function
  | Sv.Int i -> Fmt.int ppf i
  | Sv.Float f -> Format.pp_print_float ppf f
  | Sv.Bool b -> Fmt.bool ppf b

let pp_int_list ppf xs =
  Fmt.pf ppf "[%a]" (Fmt.list ~sep:(Fmt.any ",") Fmt.int) xs

let pp_opt_none pp ppf opt = Core.Pretty.option_or ~none:"none" pp ppf opt

(* One arg's value, for the "pretty print" of an op call: tensors show
   dtype+shape (never their synthesized contents), everything else shows its
   literal value. *)
let pp_arg_value ppf (av : Av.t) =
  match av with
  | Av.Tensor ts -> pp_tensor_spec ppf ts
  | Av.Tensor_opt opt -> pp_opt_none pp_tensor_spec ppf opt
  | Av.Tensor_list ts ->
      Fmt.pf ppf "[%a]" (Fmt.list ~sep:(Fmt.any "; ") pp_tensor_spec) ts
  | Av.Int i -> Fmt.int ppf i
  | Av.Int_opt opt -> pp_opt_none Fmt.int ppf opt
  | Av.Int_list xs -> pp_int_list ppf xs
  | Av.Int_list_opt opt -> pp_opt_none pp_int_list ppf opt
  | Av.Float f -> Format.pp_print_float ppf f
  | Av.Float_opt opt -> pp_opt_none Format.pp_print_float ppf opt
  | Av.Bool b -> Fmt.bool ppf b
  | Av.Bool_opt opt -> pp_opt_none Fmt.bool ppf opt
  | Av.Scalar sv -> pp_scalar_value ppf sv
  | Av.Scalar_opt opt -> pp_opt_none pp_scalar_value ppf opt
  | Av.Str s -> Fmt.pf ppf "%S" s

let pp_op_call ppf (spec : Aten_spec.Op_spec.t) =
  Fmt.pf ppf "%s(%a)" spec.target
    (Fmt.list ~sep:(Fmt.any ", ") (fun ppf (name, av) ->
         Fmt.pf ppf "%s=%a" name pp_arg_value av))
    spec.args

(* ATen-only: no native-engine comparison at all (unlike [run]/[eval_print]).
   Prints the op's pretty-printed call, each output's shape, and a status
   line; the promoted cram output is itself the golden reference (see
   .ai/pt2_node_spec_design.md) — there is no PyTorch ground truth to compare
   against. *)
let eval_report ?(ppf = Format.std_formatter) (spec : Aten_spec.Op_spec.t) :
    unit =
  let env, node = to_node spec in
  Format.fprintf ppf "[node] %a@." pp_op_call spec;
  match Interp_dispatch.dispatch env node with
  | Error e ->
      Format.fprintf ppf "  status: error: %a@." pp_interp_error
        (Err.Error.kind e)
  | Ok env' ->
      let out_names =
        List.filter_map
          (function
            | Argument.Tensor ta -> Some ta.TensorArgument.name | _ -> None)
          node.outputs
      in
      List.iter
        (fun name ->
          Format.fprintf ppf "  -> %s: %a@." name
            (Core.Pretty.option_or ~none:"<missing>" (fun ppf t ->
                 pp_shape ppf (Array.to_list (Aten_tensor.shape t))))
            (String_map.find_opt name env'))
        out_names;
      Format.fprintf ppf "  status: ok@."

(* A step's own independent seed: distinct from [Pcg.default] and from every
   other step, but still reproducible from the step index alone. *)
let walk_pcg step = Pcg.seed ~seed:(Int64.of_int step) ~seq:0xda3e39cb94b95bdbL

(* min/max/mean over every element — not the raw values, which would make a
   walk over a real model's fixtures (e.g. a 224x224x3 image, or a wide conv
   activation) unreadably large. [None] on an empty tensor. *)
let summarize_f32 (ba : Aten_tensor.float32_array) =
  let n = Bigarray.Array1.dim ba in
  if n = 0 then None
  else
    let rec go i mn mx sum =
      if i >= n then (mn, mx, sum)
      else
        let v = ba.{i} in
        go (i + 1) (Float.min mn v) (Float.max mx v) (sum +. v)
    in
    let mn, mx, sum = go 1 ba.{0} ba.{0} ba.{0} in
    Some (mn, mx, sum /. float_of_int n)

(* Same as [summarize_f32], for int64 tensors (e.g. max_pool2d_with_indices's
   index output) — kept in Int64 arithmetic throughout so min/max/sum can
   never overflow OCaml's 63-bit int, only converting to float for the final
   mean. *)
let summarize_i64 (ba : Aten_tensor.int64_array) =
  let n = Bigarray.Array1.dim ba in
  if n = 0 then None
  else
    let rec go i mn mx sum =
      if i >= n then (mn, mx, sum)
      else
        let v = ba.{i} in
        go (i + 1) (Int64.min mn v) (Int64.max mx v) (Int64.add sum v)
    in
    let mn, mx, sum = go 1 ba.{0} ba.{0} ba.{0} in
    Some (mn, mx, Int64.to_float sum /. float_of_int n)

(* 4 significant digits: enough to sanity-check a value, coarse enough to
   survive the last-ULP differences an op's SIMD reduction order can produce
   across architectures (e.g. a mean over many elements), which would
   otherwise make cram goldens flaky between machines. *)
let pp_tensor_summary ppf t =
  pp_shape ppf (Array.to_list (Aten_tensor.shape t));
  match Aten_tensor.as_float32 t with
  | Some ba -> (
      match summarize_f32 ba with
      | None -> Format.pp_print_string ppf " <empty>"
      | Some (mn, mx, mean) ->
          Format.fprintf ppf " min=%.4g max=%.4g mean=%.4g" mn mx mean)
  | None -> (
      match Aten_tensor.as_int64 t with
      | Some ba when Bigarray.Array1.dim ba < 16 ->
          Format.pp_print_char ppf ' ';
          Aten_tensor.pp_int64 ppf ba
      | Some ba -> (
          match summarize_i64 ba with
          | None -> Format.pp_print_string ppf " <empty>"
          | Some (mn, mx, mean) ->
              Format.fprintf ppf " min=%Ld max=%Ld mean=%.4g" mn mx mean)
      | None -> Format.pp_print_string ppf " <unsupported dtype>")

(* Every named tensor axis of a spec (in declared order): a "self"/"input"/
   "weight"/... arg whose contents can be resampled independently. *)
let tensor_axes (spec : Aten_spec.Op_spec.t) =
  List.filter_map
    (fun (name, av) ->
      match av with
      | Av.Tensor ts | Av.Tensor_opt (Some ts) -> Some (name, ts)
      | _ -> None)
    spec.args

(* Walks [spec] over [steps] steps. Every step mutates exactly one axis —
   one named tensor arg, round-robin over the op's tensor args in declared
   order (starting from a baseline where every axis already holds a fresh
   draw) — drawing it anew from its own distribution while every other axis
   keeps its previous value; the target/shapes/hyperparameters
   ("configuration") never change, only which axis was just redrawn. Prints
   the modified axis, the configuration, and a compact summary of the
   output. ATen-only, no native comparison (see [eval_report]). *)
let walk_eval ?(ppf = Format.std_formatter) ~steps (spec : Aten_spec.Op_spec.t)
    : unit =
  let axes = Array.of_list (tensor_axes spec) in
  let n_axes = Array.length axes in
  let env0, node = to_node ~pcg0:(walk_pcg 0) spec in
  let report ~step ~axis env =
    Format.fprintf ppf "[step %d/%d] modified axis: %s@." step steps axis;
    Format.fprintf ppf "  %a@." pp_op_call spec;
    match Interp_dispatch.dispatch env node with
    | Error e ->
        Format.fprintf ppf "  status: error: %a@." pp_interp_error
          (Err.Error.kind e)
    | Ok env' ->
        let out_names =
          List.filter_map
            (function
              | Argument.Tensor ta -> Some ta.TensorArgument.name | _ -> None)
            node.outputs
        in
        List.iter
          (fun name ->
            Format.fprintf ppf "  -> %s: %a@." name
              (Core.Pretty.option_or ~none:"<missing>" pp_tensor_summary)
              (String_map.find_opt name env'))
          out_names;
        Format.fprintf ppf "  status: ok@."
  in
  let rec go step env =
    if step <= steps then (
      if n_axes = 0 then (
        report ~step ~axis:"(none)" env;
        go (step + 1) env)
      else
        let name, ts = axes.((step - 1) mod n_axes) in
        let _, t = synthesize (walk_pcg step) ts in
        let env = String_map.add name t env in
        report ~step ~axis:name env;
        go (step + 1) env)
  in
  go 1 env0

(* Like [run] (ATen-vs-native compare), but prints the input tensors'
   shapes/distributions (the built config) and the ATen output's shape +
   min/max/mean before the status line, rather than just matched/skipped/
   mismatched with no other context. Suitable as a [Walk_core.Walk.run]
   ~verify — e.g. for walking one of the generated recipes
   (lib/aten_op_walk) with this richer per-step report instead of [run]'s
   terse one. *)
let compare_report ?(ppf = Format.std_formatter) (spec : Aten_spec.Op_spec.t) :
    bool =
  let env, node = to_node spec in
  Format.fprintf ppf "%a@." pp_op_call spec;
  match Interp_dispatch.dispatch env node with
  | Error e -> (
      match Err.Error.kind e with
      | `Unhandled_op _ ->
          Format.fprintf ppf "  status: skipped (aten interp: unhandled op)@.";
          true
      | kind ->
          Format.fprintf ppf "  status: aten interp error: %a@." pp_interp_error
            kind;
          false)
  | Ok env' -> (
      let out_names =
        List.filter_map
          (function
            | Argument.Tensor ta -> Some ta.TensorArgument.name | _ -> None)
          node.outputs
      in
      List.iter
        (fun name ->
          Format.fprintf ppf "  -> %s: %a@." name
            (Core.Pretty.option_or ~none:"<missing>" pp_tensor_summary)
            (String_map.find_opt name env'))
        out_names;
      match Op_bridge.dispatch ~aten_env:env node with
      | None ->
          Format.fprintf ppf "  status: skipped (no native impl)@.";
          true
      | Some (Error e) ->
          Format.fprintf ppf "  status: bridge error: %a@." Op_bridge.pp_error
            (Err.Error.kind e);
          false
      | Some (Ok (graph, bindings)) -> (
          match Eval_direct.run graph ~inputs:bindings with
          | Error e ->
              Format.fprintf ppf "  status: eval error: %a@."
                Eval_direct.pp_error (Err.Error.kind e);
              false
          | Ok result_env ->
              let native_outputs =
                List.map
                  (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
                  graph.Graph_ir.Graph.outputs
              in
              let atol = Verify.atol_for_target node.target in
              let errors =
                Verify.verify_node ~atol ~aten_env:env' node native_outputs
              in
              if errors = [] then (
                Format.fprintf ppf "  status: matched@.";
                true)
              else (
                Verify.report ppf node.target errors;
                false)))
