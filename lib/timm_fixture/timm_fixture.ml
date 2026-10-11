open Err.Syntax
module Index = Index
module Json = Json
module Preprocess = Preprocess
module J = Json

let ( >>= ) = Err.bind

type error = Json.error

let pp_error = Json.pp_error
let bad = J.bad

module Descriptor = struct
  type t = {
    bytes : int;
    dtype : Pt2_dtype.t;
    layout : string;
    sha256 : Pt2_sha256.Digest.t;
    shape : int list;
  }

  let dtype = function
    | "float32" -> Ok Pt2_dtype.Float32
    | "uint8" -> Ok Pt2_dtype.UInt8
    | _ -> bad "unsupported tensor dtype"

  let of_json j =
    let* bytes = J.member "bytes" j >>= J.integer in
    let* dtype = J.field "dtype" j >>= dtype in
    let* layout = J.field "layout" j in
    let* sha256 = J.field "sha256" j >>= J.digest in
    let* shape = J.member "shape" j >>= J.shape in
    let* order = J.field "byte_order" j in
    let* () = J.same "byte order" order "little" in
    let* () =
      if List.mem layout [ "CHW"; "row_major" ] then Ok ()
      else bad "unsupported layout"
    in
    let count =
      List.fold_left
        (fun n x -> if n > 134217728 / x then 134217729 else n * x)
        1 shape
    in
    let* () =
      if
        count > 134217728 || bytes > 134217728
        || bytes <> count * Pt2_dtype.element_size dtype
      then bad "tensor byte count disagrees"
      else Ok ()
    in
    Ok { bytes; dtype; layout; sha256; shape }

  let check d (tensor : Pt2_tensor.t) =
    let* () = J.same "tensor dtype" d.dtype tensor.dtype in
    let* () = J.same "tensor shape" d.shape tensor.sizes in
    let* logical =
      Pt2_fixture.Logical.of_pt2 ~max_bytes:134217728 tensor
      |> Err.map_error (fun e ->
          `Timm_fixture (Fmt.str "%a" Pt2_fixture.Logical.pp_error e))
    in
    let* () =
      J.same "tensor byte count" d.bytes (Pt2_storage.length logical.data)
    in
    let* () =
      J.same "tensor digest" d.sha256 (Pt2_sha256.bigstring logical.data)
    in
    let rec finite i =
      if i = Pt2_tensor.numel tensor || tensor.dtype = Pt2_dtype.UInt8 then
        Ok ()
      else if Float.is_finite (Pt2_fixture.Logical.get_float logical i) then
        finite (i + 1)
      else bad "nonfinite tensor value"
    in
    finite 0
end

module Binding = struct
  type transform = Identity | Select_cast | Unsqueeze

  type t = {
    graph : string;
    graph_shape : int list;
    sample : string;
    sample_layout : string;
    sample_shape : int list;
    transform : transform;
  }

  let transform ~output j =
    if output then
      let* ops = J.array j in
      match ops with
      | [] -> Ok Identity
      | [ select; cast ] ->
          let* () = J.fields [ "op"; "axis"; "index" ] select in
          let* () = J.fields [ "op"; "dtype" ] cast in
          let* op = J.field "op" select in
          let* axis = J.member "axis" select >>= J.integer in
          let* index = J.member "index" select >>= J.integer in
          let* cop = J.field "op" cast in
          let* dtype = J.field "dtype" cast in
          if (op, axis, index, cop, dtype) = ("select", 0, 0, "cast", "float32")
          then Ok Select_cast
          else bad "unsupported output transform"
      | _ -> bad "unsupported output transform"
    else
      let* op = J.field "op" j in
      match op with
      | "identity" ->
          let+ () = J.fields [ "op" ] j in
          Identity
      | "unsqueeze" ->
          let* () = J.fields [ "op"; "axis" ] j in
          let* axis = J.member "axis" j >>= J.integer in
          if axis = 0 then Ok Unsqueeze else bad "unsupported input axis"
      | _ -> bad "unsupported input transform"

  let of_json ~output j =
    let side = if output then "output" else "input" in
    let* graph = J.field ("graph_" ^ side) j in
    let* graph_shape = J.member "graph_shape" j >>= J.shape in
    let* graph_dtype = J.field "graph_dtype" j in
    let* () = J.same "graph dtype" graph_dtype "float32" in
    let* sample = J.field ("sample_" ^ side) j in
    let* metadata = J.member "sample" j in
    let* sample_shape = J.member "shape" metadata >>= J.shape in
    let* sample_dtype = J.field "dtype" metadata in
    let* () = J.same "sample dtype" sample_dtype "float32" in
    let* sample_layout = J.field "layout" metadata in
    let* () =
      if List.mem sample_layout [ "CHW"; "row_major" ] then Ok ()
      else bad "unsupported sample layout"
    in
    let* transform = J.member "transform" j >>= transform ~output in
    let transformed =
      match transform with
      | Identity -> sample_shape
      | Unsqueeze -> 1 :: sample_shape
      | Select_cast -> graph_shape
    in
    let* () =
      if transform = Select_cast then
        match graph_shape with
        | 1 :: rest -> J.same "output sample shape" rest sample_shape
        | _ -> bad "output select requires leading extent 1"
      else J.same "transformed shape" transformed graph_shape
    in
    Ok { graph; graph_shape; sample; sample_layout; sample_shape; transform }

  let input b (tensor : Pt2_tensor.t) =
    let* () = J.same "stored input shape" b.sample_shape tensor.sizes in
    let* () = J.same "stored input dtype" Pt2_dtype.Float32 tensor.dtype in
    match b.transform with
    | Identity -> Ok tensor
    | Unsqueeze ->
        Ok
          {
            tensor with
            sizes = 1 :: tensor.sizes;
            strides = Pt2_tensor.numel tensor :: tensor.strides;
          }
    | Select_cast -> bad "output transform used for input"
end

module Reference = struct
  type t = { descriptor : Descriptor.t; key : string }

  let of_json j =
    let* key = J.field "tensor_key" j in
    let+ descriptor = J.member "tensor" j >>= Descriptor.of_json in
    { descriptor; key }
end

module Case = struct
  type t = {
    geometry : Jsont.json;
    id : string;
    image : string;
    inputs : (string * Reference.t) list;
    outputs : (string * Reference.t) list;
  }

  let refs j =
    let* xs = J.members j in
    Err.List.map
      (fun (name, j) ->
        let+ r = Reference.of_json j in
        (name, r))
      xs

  let of_json j =
    let* geometry = J.member "geometry" j in
    let* id = J.field "id" j in
    let* image = J.field "decoded_image_key" j in
    let* inputs = J.member "inputs" j >>= refs in
    let+ outputs = J.member "outputs" j >>= refs in
    { geometry; id; image; inputs; outputs }

  let of_string text =
    let* j = J.parse text in
    let* () = J.version 1 j in
    let* cases = J.member "cases" j >>= J.array >>= Err.List.map of_json in
    let* () =
      if cases = [] || List.length cases > 512 then
        bad "case count outside supported limits"
      else Ok ()
    in
    let+ () = J.unique (List.map (fun c -> c.id) cases) in
    cases
end

module Contract = struct
  type t = {
    atol : float;
    decoded_sha256 : Pt2_sha256.Digest.t;
    graph_sha256 : Pt2_sha256.Digest.t;
    input_asset : string;
    inputs : Binding.t list;
    model : string;
    outputs : Binding.t list;
    rtol : float;
  }

  let of_string text =
    let* j = J.parse text in
    let* () = J.version 2 j in
    let* model = J.field "model" j in
    let* graph_sha256 =
      J.path [ "graph"; "sha256" ] j >>= J.string >>= J.digest
    in
    let* decoded_sha256 =
      J.path [ "decoded_images"; "sha256" ] j >>= J.string >>= J.digest
    in
    let* asset = J.path [ "decoded_images"; "asset" ] j >>= J.string in
    let* () = J.same "decoded asset" asset "decoded_images" in
    let* input_asset = J.path [ "inputs"; "asset" ] j >>= J.string in
    let* inputs =
      J.path [ "inputs"; "bindings" ] j
      >>= J.array
      >>= Err.List.map (Binding.of_json ~output:false)
    in
    let* outputs =
      J.path [ "outputs"; "bindings" ] j
      >>= J.array
      >>= Err.List.map (Binding.of_json ~output:true)
    in
    let* () =
      Err.List.iter
        (fun bindings ->
          let* () = if bindings = [] then bad "missing bindings" else Ok () in
          let* () = J.unique (List.map (fun b -> b.Binding.graph) bindings) in
          J.unique (List.map (fun b -> b.Binding.sample) bindings))
        [ inputs; outputs ]
    in
    let* () =
      Err.List.iter
        (fun (field, value) ->
          let* actual = J.path field j >>= J.string in
          J.same "contract reference" actual value)
        [
          ([ "graph"; "path" ], "models/" ^ model ^ "/models/model.json");
          ([ "decoded_images"; "metadata_asset" ], "image_metadata");
          ([ "cases"; "member" ], "cases.json");
          ([ "outputs"; "member" ], "outputs.pt");
          ([ "preprocessing"; "member" ], "preprocessing.json");
        ]
    in
    let* () = J.same "model input asset" input_asset (model ^ "/inputs") in
    let* () =
      Err.List.iter
        (fun (side, bindings) ->
          let* users = J.path [ "graph"; side ] j >>= J.array in
          Err.List.iter2
            ~unequal_lengths:(fun _ _ ->
              `Timm_fixture "declared signature arity disagrees")
            (fun user b ->
              let* name = J.field "name" user in
              let* dtype = J.field "dtype" user in
              let* shape = J.member "shape" user >>= J.shape in
              let* () = J.same "declared user name" name b.Binding.graph in
              let* () = J.same "declared user dtype" dtype "float32" in
              J.same "declared user shape" shape b.graph_shape)
            users bindings)
        [ ("user_inputs", inputs); ("user_outputs", outputs) ]
    in
    let* comparison = J.member "comparison" j in
    let* atol = J.member "atol" comparison >>= J.number in
    let* rtol = J.member "rtol" comparison >>= J.number in
    let* () =
      if atol < 0. || rtol < 0. then bad "negative comparison tolerance"
      else Ok ()
    in
    let* finite = J.field "finite" comparison in
    let* () = J.same "finite policy" finite "required" in
    let* rule = J.field "rule" comparison in
    let* () =
      J.same "comparison rule" rule
        "abs(actual-reference)<=atol+rtol*abs(reference)"
    in
    let* exact = J.field "preprocessing" comparison in
    let+ () = J.same "preprocessing policy" exact "exact_bits" in
    {
      atol;
      decoded_sha256;
      graph_sha256;
      input_asset;
      inputs;
      model;
      outputs;
      rtol;
    }

  let check_graph t ~graph_bytes program =
    let open Pytorch_types in
    let signature =
      program.ExportedProgram.graph_module.GraphModule.signature
    in
    let* () =
      J.same "graph digest" t.graph_sha256 (Pt2_sha256.string graph_bytes)
    in
    let input_names =
      List.filter_map
        (function
          | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor a } ->
              Some a.TensorArgument.name
          | InputSpec.User_input _ -> Some "<unsupported-scalar>"
          | _ -> None)
        signature.GraphSignature.input_specs
    in
    let output_names =
      List.filter_map
        (function
          | OutputSpec.User_output { UserOutputSpec.arg = Argument.Tensor a } ->
              Some a.TensorArgument.name
          | OutputSpec.User_output _ -> Some "<unsupported-scalar>"
          | _ -> None)
        signature.GraphSignature.output_specs
    in
    let* () =
      J.same "input binding coverage/order" input_names
        (List.map (fun b -> b.Binding.graph) t.inputs)
    in
    let* () =
      J.same "output binding coverage/order" output_names
        (List.map (fun b -> b.Binding.graph) t.outputs)
    in
    let values = program.graph_module.graph.tensor_values in
    Err.List.iter
      (fun b ->
        let* meta =
          Err.of_option (`Timm_fixture "missing graph tensor metadata")
            (Schema_runtime.String_map.find_opt b.Binding.graph values)
        in
        let* shape =
          Err.List.map
            (fun x ->
              Pt2_tensor.int_of_symint x
              |> Err.map_error (fun _ ->
                  `Timm_fixture "dynamic shape unsupported"))
            meta.TensorMeta.sizes
        in
        let* () = J.same "graph tensor shape" shape b.graph_shape in
        J.same "graph tensor dtype" meta.dtype ScalarType.FLOAT)
      (t.inputs @ t.outputs)
end

let bind_inputs contract case tensors =
  let* () =
    J.same "case input coverage"
      (List.sort String.compare (List.map fst case.Case.inputs))
      (List.sort String.compare
         (List.map (fun b -> b.Binding.sample) contract.Contract.inputs))
  in
  Err.List.map
    (fun binding ->
      let reference = List.assoc binding.Binding.sample case.inputs in
      let* tensor =
        Err.of_option
          (`Timm_fixture ("missing tensor key " ^ reference.Reference.key))
          (List.assoc_opt reference.key tensors)
      in
      let* () = Descriptor.check reference.descriptor tensor in
      let* () =
        J.same "sample layout" reference.descriptor.layout binding.sample_layout
      in
      let+ tensor = Binding.input binding tensor in
      (binding.graph, tensor))
    contract.inputs

let compare_output contract binding reference actual =
  let* () =
    J.same "actual output dtype" actual.Pt2_fixture.Logical.dtype
      Pt2_checkpoint_map.Dtype.F32
  in
  let* () =
    J.same "actual graph output shape" actual.shape
      (List.map Int64.of_int binding.Binding.graph_shape)
  in
  let shape =
    match binding.transform with
    | Binding.Select_cast -> List.tl binding.graph_shape
    | Identity -> binding.graph_shape
    | Unsqueeze -> []
  in
  let* () = J.same "reference output shape" reference.Pt2_tensor.sizes shape in
  let* logical =
    Pt2_fixture.Logical.of_pt2 reference
    |> Err.map_error (fun e ->
        `Timm_fixture (Fmt.str "%a" Pt2_fixture.Logical.pp_error e))
  in
  let rec go i =
    if i = Pt2_tensor.numel reference then Ok ()
    else
      let a = Pt2_fixture.Logical.get_float actual i
      and r = Pt2_fixture.Logical.get_float logical i in
      if
        Float.is_finite a && Float.is_finite r
        && Float.abs (a -. r)
           <= contract.Contract.atol +. (contract.rtol *. Float.abs r)
      then go (i + 1)
      else bad (Printf.sprintf "output element %d: %.9g vs %.9g" i a r)
  in
  go 0

let check_references bindings references tensors =
  let* () =
    J.same "case reference coverage"
      (List.sort String.compare (List.map fst references))
      (List.sort String.compare (List.map (fun b -> b.Binding.sample) bindings))
  in
  Err.List.iter
    (fun b ->
      let r = List.assoc b.Binding.sample references in
      let* t =
        Err.of_option
          (`Timm_fixture ("missing tensor key " ^ r.Reference.key))
          (List.assoc_opt r.key tensors)
      in
      let* () = J.same "reference shape" r.descriptor.shape b.sample_shape in
      let* () = J.same "reference layout" r.descriptor.layout b.sample_layout in
      let* () = J.same "reference dtype" r.descriptor.dtype Pt2_dtype.Float32 in
      Descriptor.check r.descriptor t)
    bindings

let check_cases contract cases ~inputs ~outputs =
  let* () =
    Err.List.iter
      (fun c ->
        let* () =
          check_references contract.Contract.inputs c.Case.inputs inputs
        in
        check_references contract.outputs c.outputs outputs)
      cases
  in
  Err.List.iter
    (fun (refs, tensors) ->
      let used =
        List.concat_map
          (fun c -> List.map (fun (_, r) -> r.Reference.key) (refs c))
          cases
        |> List.sort_uniq String.compare
      in
      J.same "tensor key coverage" used
        (List.sort String.compare (List.map fst tensors)))
    [ ((fun c -> c.Case.inputs), inputs); ((fun c -> c.Case.outputs), outputs) ]

module Images = struct
  let check ~index ~metadata tensors =
    let* () = J.version 1 metadata in
    let* decoded = Index.asset_of index "decoded_images" in
    let* legacy = Index.asset_of index "legacy_images" in
    let* actual = J.field "images_sha256" metadata >>= J.digest in
    let* () = J.same "decoded metadata digest" actual decoded.pin.sha256 in
    let* actual = J.field "source_archive_sha256" metadata >>= J.digest in
    let* () = J.same "source archive identity" actual legacy.pin.sha256 in
    let* () =
      Err.List.iter
        (fun (field, value) ->
          let* actual = J.path [ "decoding"; field ] metadata >>= J.string in
          J.same field actual value)
        [ ("color_mode", "RGB"); ("exif_orientation", "not_applied") ]
    in
    let* samples = J.member "samples" metadata >>= J.members in
    let* order =
      J.member "sample_order" metadata >>= J.array >>= Err.List.map J.string
    in
    let* () =
      J.same "image order" order
        (List.sort String.compare (List.map fst samples))
    in
    let* () =
      J.same "decoded image key coverage" order
        (List.sort String.compare (List.map fst tensors))
    in
    Err.List.iter
      (fun (key, j) ->
        let* _ = J.field "source_sha256" j >>= J.digest in
        let* d = J.member "tensor" j >>= Descriptor.of_json in
        let* () = J.same "image layout" d.layout "CHW" in
        let* () = J.same "image dtype" d.dtype Pt2_dtype.UInt8 in
        let* () =
          match d.shape with
          | [ 3; _; _ ] -> Ok ()
          | _ -> bad "decoded image is not RGB CHW"
        in
        Descriptor.check d (List.assoc key tensors))
      samples
end
