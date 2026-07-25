(* Derives one Aten_spec.Op_spec.t JSON per node of a real exported .pt2
   model graph: the node's real target and hyperparameters are kept verbatim,
   but every tensor argument's *contents* are replaced by a synthesized fill —
   never the trained weight/buffer values themselves. Where a tensor argument
   resolves to a real parameter/buffer (via the graph's signature), the
   distribution is fit to that tensor's own sample statistics (mean/variance,
   or min/max when the real data is always positive); true intermediate
   activations have no recorded real data (each node is tested standalone, not
   by running the full network), so they fall back to a fixed Normal(0,1).

   Errors are threaded as Core.result throughout (matching Pt2_dtype/
   Pt2_tensor/Pt2_archive's own error framework), not converted to
   exceptions; only [write_dir], the outermost driver, pattern-matches the
   result per node to decide whether to write a file or skip with a
   warning. *)

open Pytorch_types
open Schema_runtime
open Core.Syntax

module Stats = struct
  type t = { min : float; max : float; mean : float; variance : float }
end

type error =
  [ Pt2_tensor.error
  | Pt2_archive.error
  | `Unsupported_pt2_dtype of Pt2_dtype.t
  | `Non_contiguous_tensor
  | `No_tensor_meta of string
  | `Missing_arg of string
  | `Unexpected_arg of string
  | `Unbound_op of string ]

let pp_error ppf : error -> unit = function
  | #Pt2_tensor.error as e -> Pt2_tensor.pp_error ppf e
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | `Unsupported_pt2_dtype d ->
      Format.fprintf ppf "unsupported dtype %s" (Pt2_dtype.to_string d)
  | `Non_contiguous_tensor ->
      Format.fprintf ppf "cannot compute stats of a non-contiguous tensor"
  | `No_tensor_meta name ->
      Format.fprintf ppf "no tensor_values entry for %S" name
  | `Missing_arg name -> Format.fprintf ppf "missing required arg %S" name
  | `Unexpected_arg name ->
      Format.fprintf ppf "unexpected value for arg %S" name
  | `Unbound_op target -> Format.fprintf ppf "unbound op %S" target

let dtype_of_pt2 : Pt2_dtype.t -> (Aten_spec.Dtype.t, [> error ]) Core.result =
  function
  | Pt2_dtype.Float32 -> Core.return Aten_spec.Dtype.F32
  | Pt2_dtype.Float64 -> Core.return Aten_spec.Dtype.F64
  | Pt2_dtype.Int64 -> Core.return Aten_spec.Dtype.I64
  | Pt2_dtype.Int32 -> Core.return Aten_spec.Dtype.I32
  | Pt2_dtype.Bool -> Core.return Aten_spec.Dtype.Bool
  | (Pt2_dtype.Int16 | Pt2_dtype.Int8 | Pt2_dtype.UInt8) as d ->
      Core.fail (`Unsupported_pt2_dtype d)

let dtype_of_scalar_type st =
  let* d =
    Pt2_dtype.of_scalar_type st |> Core.map_error (fun e -> (e :> error))
  in
  dtype_of_pt2 d

let shape_of (m : TensorMeta.t) =
  Core.List.map
    (fun si ->
      Pt2_tensor.int_of_symint si |> Core.map_error (fun e -> (e :> error)))
    m.sizes

(* Sample statistics over every element of a real, contiguous tensor. Only
   f32 is expected for the models this targets (conv/linear weights, batch
   norm stats); other dtypes are not yet needed. *)
let stats (t : Pt2_tensor.t) : (Stats.t, [> error ]) Core.result =
  if not (Pt2_tensor.is_contiguous t) then Core.fail `Non_contiguous_tensor
  else
    match t.dtype with
    | Pt2_dtype.Float32 ->
        let data = t.data in
        let n = Pt2_tensor.numel t in
        let get i = Int32.float_of_bits (Bytes.get_int32_le data (i * 4)) in
        let v0 = get 0 in
        let rec scan i min max sum =
          if i >= n then (min, max, sum)
          else
            let v = get i in
            scan (i + 1) (Float.min min v) (Float.max max v) (sum +. v)
        in
        let min, max, sum = scan 1 v0 v0 v0 in
        let mean = sum /. float_of_int n in
        let rec sum_sq i acc =
          if i >= n then acc
          else
            let d = get i -. mean in
            sum_sq (i + 1) (acc +. (d *. d))
        in
        let variance = sum_sq 0 0. /. float_of_int n in
        Core.return { Stats.min; max; mean; variance }
    | d -> Core.fail (`Unsupported_pt2_dtype d)

(* The tensor's own real data picks its distribution: values that are always
   positive (e.g. batch_norm's running_var, which must stay positive since
   batch_norm computes 1/sqrt(running_var+eps)) get a Uniform spanning the
   real range; anything else gets a Normal fit to the real mean/variance (a
   small variance floor guards a degenerate all-equal tensor, e.g. a
   zero-initialized bias). *)
let distribution_of_stats (s : Stats.t) : Aten_spec.Distribution.t =
  if s.min > 0. then Uniform { low = s.min; high = s.max }
  else Normal { mean = s.mean; variance = Float.max s.variance 1e-6 }

(* No real data is available for a true intermediate activation (a previous
   node's output in the real graph) since each node is tested standalone. *)
let default_activation_distribution : Aten_spec.Distribution.t =
  Normal { mean = 0.; variance = 1. }

(* Graph-input SSA name -> captured payload name, for every inference tensor
   source. Mirrors lib/interp/interp.ml's mapping (this generator stays pure). *)
let params_of_signature (sign : GraphSignature.t) : string String_map.t =
  List.fold_left
    (fun acc (spec : InputSpec.t) ->
      match spec with
      | InputSpec.Parameter p ->
          String_map.add p.InputToParameterSpec.arg.name
            p.InputToParameterSpec.parameter_name acc
      | InputSpec.Buffer b ->
          String_map.add b.InputToBufferSpec.arg.name
            b.InputToBufferSpec.buffer_name acc
      | InputSpec.Tensor_constant c ->
          String_map.add c.InputToTensorConstantSpec.arg.name
            c.InputToTensorConstantSpec.tensor_constant_name acc
      | _ -> acc)
    String_map.empty sign.input_specs

let tensor_spec_of archive params tensor_values name :
    (Aten_spec.Tensor_spec.t, [> error ]) Core.result =
  match String_map.find_opt name params with
  | Some cfg_name ->
      let* wt =
        Pt2_archive.load_captured_tensor archive cfg_name
        |> Core.map_error (fun e -> (e :> error))
      in
      let* dtype = dtype_of_pt2 wt.dtype in
      let* s = stats wt in
      Core.return
        {
          Aten_spec.Tensor_spec.dtype;
          shape = wt.sizes;
          source = Random (distribution_of_stats s);
        }
  | None -> (
      match String_map.find_opt name tensor_values with
      | Some (meta : TensorMeta.t) ->
          let* dtype = dtype_of_scalar_type meta.dtype in
          let* shape = shape_of meta in
          Core.return
            {
              Aten_spec.Tensor_spec.dtype;
              shape;
              source = Random default_activation_distribution;
            }
      | None -> Core.fail (`No_tensor_meta name))

let find_named_arg (node : Node.t) name =
  List.find_map
    (fun (na : NamedArgument.t) ->
      if String.equal na.name name then Some na.arg else None)
    node.inputs

(* One op-config param -> its (name, Arg_value) pair, decoded from the node's
   real argument (falling back to the schema default when absent — real
   exported graphs list every arg explicitly, so this is mostly a safety
   net). [None] means the key is dropped from the JSON entirely, not a
   failure: either the param is a ScalarType?/Layout?/MemoryFormat?/Device?
   (same as the generated per-op specs — see .ai/aten_op_config_design.md),
   or it's a genuinely-absent `_opt` arg — the generated codecs decode those
   with [Jsont.Object.opt_mem] (absent key -> None), which does not accept an
   explicit JSON null in the value's place. *)
let arg_value_of_node ~tensor_spec (node : Node.t) (p : Aten_op_config.param) :
    ((string * Aten_spec.Arg_value.t) option, [> error ]) Core.result =
  let name = p.name in
  let found = find_named_arg node name in
  let module Av = Aten_spec.Arg_value in
  let module Sv = Aten_spec.Scalar_value in
  let missing () = Core.fail (`Missing_arg name) in
  let unexpected () = Core.fail (`Unexpected_arg name) in
  let some av = Core.return (Some (name, av)) in
  match p.ty with
  | Aten_op_config.Tensor -> (
      match found with
      | Some (Argument.Tensor ta) ->
          let* ts = tensor_spec ta.name in
          some (Av.Tensor ts)
      | _ -> unexpected ())
  | Aten_op_config.Tensor_opt -> (
      match found with
      | Some (Argument.Tensor ta)
      | Some (Argument.Optional_tensor (OptionalTensorArgument.Tensor ta)) ->
          let* ts = tensor_spec ta.name in
          some (Av.Tensor_opt (Some ts))
      | Some (Argument.None _)
      | Some (Argument.Optional_tensor (OptionalTensorArgument.None _))
      | None ->
          (* The generated per-op codecs decode `_opt` fields with
             [Jsont.Object.opt_mem]: absent key -> None. A present-but-null
             value is *not* equivalent — opt_mem still hands it to the value
             codec, which then fails to decode Tensor_spec/etc. from null. So
             an absent optional arg must drop the key entirely, not encode an
             explicit `_opt None`. *)
          Core.return None
      | _ -> unexpected ())
  | Aten_op_config.Tensor_list -> (
      match found with
      | Some (Argument.Tensors tas) ->
          let* specs =
            Core.List.map
              (fun (ta : TensorArgument.t) -> tensor_spec ta.name)
              tas
          in
          some (Av.Tensor_list specs)
      | _ -> unexpected ())
  | Aten_op_config.Int -> (
      match (found, p.default) with
      | Some (Argument.Int i), _ -> some (Av.Int i)
      | None, Some (Aten_op_config.D_int i) -> some (Av.Int i)
      | None, _ -> missing ()
      | _ -> unexpected ())
  | Aten_op_config.Int_opt -> (
      match found with
      | Some (Argument.Int i) -> some (Av.Int_opt (Some i))
      | Some (Argument.None _) | None -> Core.return None
      | _ -> unexpected ())
  | Aten_op_config.Int_list -> (
      match (found, p.default) with
      | Some (Argument.Ints xs), _ -> some (Av.Int_list xs)
      | Some (Argument.Int i), _ -> some (Av.Int_list [ i ])
      | None, Some (Aten_op_config.D_ints xs) -> some (Av.Int_list xs)
      | None, Some (Aten_op_config.D_int i) -> some (Av.Int_list [ i ])
      | None, _ -> missing ()
      | _ -> unexpected ())
  | Aten_op_config.Int_list_opt -> (
      match found with
      | Some (Argument.Ints xs) -> some (Av.Int_list_opt (Some xs))
      | Some (Argument.None _) | None -> Core.return None
      | _ -> unexpected ())
  | Aten_op_config.Float -> (
      match (found, p.default) with
      | Some (Argument.Float f), _ -> some (Av.Float f)
      | None, Some (Aten_op_config.D_float f) -> some (Av.Float f)
      | None, _ -> missing ()
      | _ -> unexpected ())
  | Aten_op_config.Float_opt -> (
      match found with
      | Some (Argument.Float f) -> some (Av.Float_opt (Some f))
      | Some (Argument.None _) | None -> Core.return None
      | _ -> unexpected ())
  | Aten_op_config.Bool -> (
      match (found, p.default) with
      | Some (Argument.Bool b), _ -> some (Av.Bool b)
      | None, Some (Aten_op_config.D_bool b) -> some (Av.Bool b)
      | None, _ -> missing ()
      | _ -> unexpected ())
  | Aten_op_config.Bool_opt -> (
      match found with
      | Some (Argument.Bool b) -> some (Av.Bool_opt (Some b))
      | Some (Argument.None _) | None -> Core.return None
      | _ -> unexpected ())
  | Aten_op_config.Scalar -> (
      match (found, p.default) with
      | Some (Argument.Int i), _ -> some (Av.Scalar (Sv.Int i))
      | Some (Argument.Float f), _ -> some (Av.Scalar (Sv.Float f))
      | Some (Argument.Bool b), _ -> some (Av.Scalar (Sv.Bool b))
      | None, Some (Aten_op_config.D_int i) -> some (Av.Scalar (Sv.Int i))
      | None, Some (Aten_op_config.D_float f) -> some (Av.Scalar (Sv.Float f))
      | None, _ -> missing ()
      | _ -> unexpected ())
  | Aten_op_config.Scalar_opt -> (
      match found with
      | Some (Argument.Int i) -> some (Av.Scalar_opt (Some (Sv.Int i)))
      | Some (Argument.Float f) -> some (Av.Scalar_opt (Some (Sv.Float f)))
      | Some (Argument.None _) | None -> Core.return None
      | _ -> unexpected ())
  | Aten_op_config.Str -> (
      match (found, p.default) with
      | Some (Argument.String s), _ -> some (Av.Str s)
      | None, Some (Aten_op_config.D_str s) -> some (Av.Str s)
      | None, _ -> missing ()
      | _ -> unexpected ())
  | Aten_op_config.Scalar_type_opt | Aten_op_config.Layout_opt
  | Aten_op_config.Memory_format_opt | Aten_op_config.Device_opt ->
      Core.return None

let spec_of_node archive params tensor_values (node : Node.t) :
    (Aten_spec.Op_spec.t, [> error ]) Core.result =
  match Aten_op_config.find node.target with
  | None -> Core.fail (`Unbound_op node.target)
  | Some cfg ->
      let tensor_spec = tensor_spec_of archive params tensor_values in
      let* args_rev =
        Core.List.fold_left
          (fun acc p ->
            let* item = arg_value_of_node ~tensor_spec node p in
            Core.return (match item with Some kv -> kv :: acc | None -> acc))
          [] cfg.params
      in
      Core.return
        { Aten_spec.Op_spec.target = node.target; args = List.rev args_rev }

(* "torch.ops.aten.convolution.default" -> "convolution_default" *)
let safe_name target =
  let target =
    match String.length target with
    | n when n > 15 && String.sub target 0 15 = "torch.ops.aten." ->
        String.sub target 15 (n - 15)
    | _ -> target
  in
  String.map (function '.' -> '_' | c -> c) target

let file_name ~idx (node : Node.t) =
  Printf.sprintf "%03d_%s.json" idx (safe_name node.target)

let encode_exn spec =
  match Jsont_bytesrw.encode_string Aten_op_spec.jsont spec with
  | Ok s -> s
  | Error e -> failwith ("pt2_spec_gen: encode: " ^ e)

let write_dir ~out_dir (archive : Pt2_archive.t) =
  let program = Pt2_archive.program archive in
  let gm = program.graph_module in
  let params = params_of_signature gm.signature in
  let g = gm.graph in
  if not (Sys.file_exists out_dir) then Sys.mkdir out_dir 0o755;
  let written, skipped =
    List.fold_left
      (fun (written, skipped) (idx, node) ->
        match spec_of_node archive params g.tensor_values node with
        | Ok spec ->
            let path = Filename.concat out_dir (file_name ~idx node) in
            let oc = open_out path in
            output_string oc (encode_exn spec);
            close_out oc;
            (written + 1, skipped)
        | Error e ->
            Printf.eprintf "pt2_spec_gen: skipping node %d (%s): %s\n" idx
              node.target
              (Format.asprintf "%a" (Core.Error.pp pp_error) e);
            (written, skipped + 1))
      (0, 0)
      (List.mapi (fun idx node -> (idx, node)) g.nodes)
  in
  (written, skipped, List.length g.nodes)
