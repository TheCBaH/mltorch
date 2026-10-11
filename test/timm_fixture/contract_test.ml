module F = Timm_fixture
module J = F.Json
module U = Timm_fixture_unix

let get r = Err.or_raise ~pp_error:F.pp_error r
let checks = ref 0

let accept r =
  ignore (get r);
  incr checks

let reject r =
  match Err.payload r with Error _ -> incr checks | Ok _ -> assert false

let parse s = get (J.parse s)
let digest s = Pt2_sha256.string s

let raw values =
  let b = Bytes.create (4 * List.length values) in
  List.iteri
    (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v))
    values;
  Bytes.to_string b

let tensor values =
  {
    Pt2_tensor.dtype = Pt2_dtype.Float32;
    sizes = [ List.length values ];
    strides = [ 1 ];
    storage_offset = 0;
    data = Pt2_storage.of_string (raw values);
  }

let descriptor values =
  {
    F.Descriptor.bytes = List.length values * 4;
    dtype = Pt2_dtype.Float32;
    layout = "row_major";
    sha256 = digest (raw values);
    shape = [ List.length values ];
  }

let binding graph sample =
  {
    F.Binding.graph;
    graph_shape = [ 2 ];
    sample;
    sample_shape = [ 2 ];
    sample_layout = "row_major";
    transform = F.Binding.Identity;
  }

let contract inputs =
  {
    F.Contract.atol = 1e-4;
    decoded_sha256 = digest "images";
    graph_sha256 = digest "graph";
    input_asset = "test/inputs";
    inputs;
    model = "test";
    outputs = [ binding "out" "logits" ];
    rtol = 1e-3;
  }

let reference key = { F.Reference.key; descriptor = descriptor [ 1.; 2. ] }

let case inputs =
  {
    F.Case.geometry = parse "[]";
    id = "case";
    image = "image";
    inputs;
    outputs = [ ("logits", reference "out") ];
  }

let meta =
  {|{"dtype":7,"sizes":[{"as_int":2}],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}

let arg n = Printf.sprintf {|{"as_tensor":{"name":%S}}|} n

let graph =
  Printf.sprintf
    {|{"graph_module":{"graph":{"inputs":[%s,%s,%s],"outputs":[%s],"nodes":[],"tensor_values":{"weight":%s,"a":%s,"z":%s,"out":%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[{"parameter":{"arg":{"name":"weight"},"parameter_name":"weight"}},{"user_input":{"arg":%s}},{"user_input":{"arg":%s}}],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":20}}|}
    (arg "weight") (arg "z") (arg "a") (arg "out") meta meta meta meta (arg "z")
    (arg "a") (arg "out")

let () =
  reject (J.parse {|{"x":1,"x":2}|});
  reject (J.parse {|{"nested":{"x":1,"x":2}}|});
  reject (J.shape (parse "[0]"));
  reject (J.shape (parse "[536870913]"));
  let binding_json ~output ~graph_shape ~sample_shape ~graph_dtype ~sample_dtype
      ~transform =
    let side = if output then "output" else "input" in
    parse
      (Printf.sprintf
         {|{"graph_%s":"graph","graph_shape":%s,"graph_dtype":%S,"sample_%s":"sample","sample":{"shape":%s,"dtype":%S,"layout":"row_major"},"transform":%s}|}
         side graph_shape graph_dtype side sample_shape sample_dtype transform)
  in
  let binding_check ~output ~graph_shape ~sample_shape ~graph_dtype
      ~sample_dtype ~transform =
    F.Binding.of_json ~output
      (binding_json ~output ~graph_shape ~sample_shape ~graph_dtype
         ~sample_dtype ~transform)
  in
  let input =
    binding_check ~output:false ~graph_shape:"[1,2]" ~sample_shape:"[2]"
      ~transform:{|{"op":"unsqueeze","axis":0}|}
  in
  accept (input ~graph_dtype:"float32" ~sample_dtype:"float32");
  reject (input ~graph_dtype:"float64" ~sample_dtype:"float32");
  reject (input ~graph_dtype:"float32" ~sample_dtype:"uint8");
  reject
    (binding_check ~output:false ~graph_shape:"[1,3]" ~sample_shape:"[2]"
       ~graph_dtype:"float32" ~sample_dtype:"float32"
       ~transform:{|{"op":"unsqueeze","axis":0}|});
  let output =
    binding_check ~output:true ~graph_dtype:"float32"
      ~transform:
        {|[{"op":"select","axis":0,"index":0},{"op":"cast","dtype":"float32"}]|}
  in
  accept
    (output ~graph_shape:"[1,2]" ~sample_shape:"[2]" ~sample_dtype:"float32");
  reject
    (output ~graph_shape:"[2,2]" ~sample_shape:"[2]" ~sample_dtype:"float32");
  reject
    (output ~graph_shape:"[1,2]" ~sample_shape:"[3]" ~sample_dtype:"float32");
  reject (output ~graph_shape:"[1,2]" ~sample_shape:"[2]" ~sample_dtype:"uint8");
  let descriptor_json dtype order shape bytes =
    parse
      (Printf.sprintf
         {|{"bytes":%d,"dtype":%S,"layout":"row_major","sha256":%S,"shape":%s,"byte_order":%S}|}
         bytes dtype (String.make 64 'a') shape order)
  in
  reject (F.Descriptor.of_json (descriptor_json "float32" "big" "[2]" 8));
  reject (F.Descriptor.of_json (descriptor_json "float64" "little" "[2]" 16));
  reject
    (F.Descriptor.of_json
       (descriptor_json "float32" "little" "[536870912,536870912]" 8));
  (match Err.payload (J.member "required" (parse "{}")) with
  | Error (`Timm_fixture (J.Failure.Missing_field "required")) -> incr checks
  | _ -> assert false);
  reject
    (F.Binding.transform ~output:false (parse {|{"op":"unsqueeze","axis":1}|}));
  reject
    (F.Binding.transform ~output:false (parse {|{"op":"identity","axis":0}|}));
  reject
    (F.Binding.transform ~output:true
       (parse
          {|[{"op":"select","axis":0,"index":1},{"op":"cast","dtype":"float32"}]|}));
  reject
    (F.Binding.transform ~output:true
       (parse {|[{"op":"cast","dtype":"float64"}]|}));
  accept (F.Descriptor.check (descriptor [ 1.; -0. ]) (tensor [ 1.; -0. ]));
  reject (F.Descriptor.check (descriptor [ 1.; -0. ]) (tensor [ 1.; 0. ]));
  reject (F.Descriptor.check (descriptor [ nan ]) (tensor [ nan ]));
  let strided =
    {
      (tensor [ 9.; 1.; 9.; 2.; 9. ]) with
      sizes = [ 2 ];
      strides = [ 2 ];
      storage_offset = 1;
    }
  in
  accept (F.Descriptor.check (descriptor [ 1.; 2. ]) strided);
  reject
    (F.Descriptor.check
       (descriptor [ 1.; 2. ])
       { strided with strides = [ 20 ] });
  let bindings = [ binding "z" "second"; binding "a" "first" ] in
  let c =
    case [ ("first", reference "a-key"); ("second", reference "z-key") ]
  in
  let stored = [ ("a-key", tensor [ 1.; 2. ]); ("z-key", tensor [ 1.; 2. ]) ] in
  let bound = get (F.bind_inputs (contract bindings) c stored) in
  assert (List.map fst bound = [ "z"; "a" ]);
  incr checks;
  reject
    (F.bind_inputs (contract bindings)
       (case [ ("first", reference "a-key") ])
       stored);
  reject (F.bind_inputs (contract bindings) c [ ("z-key", tensor [ 1.; 2. ]) ]);
  let program =
    get
      (Pt2_archive.program_of_json graph
      |> U.wrap J.External.Source.Archive Pt2_archive.pp_error)
  in
  let ct = { (contract bindings) with graph_sha256 = digest graph } in
  accept (F.Contract.check_graph ct ~graph_bytes:graph program);
  reject
    (F.Contract.check_graph
       { ct with inputs = List.rev bindings }
       ~graph_bytes:graph program);
  reject
    (F.Contract.check_graph
       { ct with inputs = [ List.hd bindings ] }
       ~graph_bytes:graph program);
  reject
    (F.Contract.check_graph
       { ct with inputs = bindings @ [ binding "weight" "capture" ] }
       ~graph_bytes:graph program);
  reject
    (F.Contract.check_graph
       { ct with inputs = [ binding "z" "second"; binding "z" "first" ] }
       ~graph_bytes:graph program);
  reject (F.Contract.check_graph ct ~graph_bytes:(graph ^ " ") program);
  accept
    (F.check_cases ct [ c ] ~inputs:stored
       ~outputs:[ ("out", tensor [ 1.; 2. ]) ]);
  reject
    (F.check_cases ct [ c ]
       ~inputs:(("unused", tensor [ 1.; 2. ]) :: stored)
       ~outputs:[ ("out", tensor [ 1.; 2. ]) ]);
  let logical values =
    get
      (Pt2_fixture.Logical.of_pt2 (tensor values)
      |> U.wrap J.External.Source.Logical Pt2_fixture.Logical.pp_error)
  in
  accept
    (F.compare_output ct (binding "out" "logits")
       (tensor [ 1.; 2. ])
       (logical [ 1.0001; 2.0001 ]));
  reject
    (F.compare_output ct (binding "out" "logits")
       (tensor [ 1.; 2. ])
       (logical [ 1.1; 2. ]));
  reject
    (F.compare_output ct (binding "out" "logits")
       (tensor [ 1.; 2. ])
       (logical [ nan; 2. ]));
  reject
    (F.compare_output ct (binding "out" "logits")
       (tensor [ 1.; 2. ])
       { (logical [ 1.; 2. ]) with shape = [ 1L; 2L ] });
  let root = Filename.temp_file "timm-cache-test" "" in
  Sys.remove root;
  Unix.mkdir root 0o700;
  let rec remove path =
    match (Unix.lstat path).st_kind with
    | Unix.S_DIR ->
        Array.iter
          (fun name -> remove (Filename.concat path name))
          (Sys.readdir path);
        Unix.rmdir path
    | _ -> Sys.remove path
  in
  Fun.protect
    ~finally:(fun () -> remove root)
    (fun () ->
      let calls = ref 0 and transferred = ref 0L in
      let transport ~url:_ ~dest =
        incr calls;
        Out_channel.with_open_bin dest (fun oc -> output_string oc "fixture");
        Ok ()
      in
      let config =
        get
          (U.Config.create ~transport
             ~on_transfer:(fun _ bytes ->
               transferred := Int64.add !transferred bytes)
             root)
      in
      let pin =
        {
          Pt2_checkpoint_map.Document.Pin.name = "asset";
          sha256 = digest "fixture";
          size = 7L;
          url = "https://example.org/asset";
        }
      in
      let ensure = U.ensure config ~layer:Pt2_fixture.Fault.Archive in
      let path = get (ensure pin) in
      accept (ensure pin);
      assert (!calls = 1 && !transferred = 7L);
      incr checks;
      Out_channel.with_open_bin path (fun oc -> output_string oc "changed");
      let offline = get (U.Config.create root) in
      reject (U.ensure offline ~layer:Pt2_fixture.Fault.Archive pin);
      accept (ensure pin);
      assert (!calls = 2);
      incr checks;
      let bad_pin = { pin with sha256 = digest "invalid" } in
      reject (ensure bad_pin);
      assert (
        not
          (Sys.file_exists
             (Pt2_fixture_unix.Cache.blob_path config.cache bad_pin.sha256)));
      assert (Sys.readdir (Pt2_fixture_unix.Cache.temp_dir config.cache) = [||]);
      incr checks);
  let commit = String.make 40 'a' in
  let asset id filename format =
    Printf.sprintf
      {|%S:{"filename":%S,"format":%S,"members":[],"bytes":7,"sha256":%S,"url":%S}|}
      id filename format
      (Pt2_sha256.Digest.to_hex (digest "fixture"))
      ("https://github.com/o/r/releases/download/t/" ^ filename)
  in
  let manifest =
    Printf.sprintf
      {|{"schema_version":2,"scope":"full","producer":{"repo":"o/r","tag":"t","commit":%S},"assets":{%s},"models":{"test":{"status":"current","archive":"test/archive","evaluation":"test/evaluation","inputs":"test/inputs","graph":{"bytes":7,"sha256":%S,"url":%S},"weight_map":null}},"release_models":{"count":1,"names":["test"]}}|}
      commit
      (String.concat ","
         [
           asset "decoded_images" "images.pt" "torch.save";
           asset "image_metadata" "images-metadata.json" "json";
           asset "legacy_images" "images.zip" "zip";
           asset "test/archive" "test.zip" "zip";
           asset "test/evaluation" "test-eval.zip" "zip";
           asset "test/inputs" "test-inputs.pt" "torch.save";
         ])
      (Pt2_sha256.Digest.to_hex (digest "fixture"))
      ("https://raw.githubusercontent.com/o/r/" ^ commit
     ^ "/models/test/models/model.json")
  in
  accept (F.Index.of_string manifest);
  accept (U.Discovery.classify ~status:"200" manifest);
  accept (U.Discovery.classify ~status:"404" "absent");
  reject (U.Discovery.classify ~status:"500" "unavailable");
  reject (U.Discovery.classify ~status:"200" "corrupt");
  let replace from replacement text =
    let n = String.length from in
    let rec go i =
      if i + n > String.length text then assert false
      else if String.sub text i n = from then
        String.sub text 0 i ^ replacement
        ^ String.sub text (i + n) (String.length text - i - n)
      else go (i + 1)
    in
    go 0
  in
  reject
    (F.Index.of_string
       (replace "\"schema_version\":2" "\"schema_version\":3" manifest));
  reject
    (U.Discovery.classify ~status:"200"
       (replace "\"schema_version\":2" "\"schema_version\":3" manifest));
  reject
    (F.Index.of_string
       (replace "\"inputs\":\"test/inputs\"" "\"inputs\":\"other/inputs\""
          manifest));
  reject
    (F.Index.of_string (replace "test-inputs.pt" "../test-inputs.pt" manifest));
  reject (F.Index.of_string (replace "\"count\":1" "\"count\":2" manifest));
  reject
    (F.Index.of_string
       (replace "raw.githubusercontent.com/o/r/"
          "raw.githubusercontent.com/other/r/" manifest));
  reject
    (F.Index.of_string
       (replace "\"format\":\"torch.save\"" "\"format\":\"pickle\"" manifest));
  Printf.printf "%d contract, tensor, discovery and cache checks passed\n"
    !checks
