(* A complete synthetic release for replay tests. *)

module W = Pt2_fixture_test.World

let jstr = Printf.sprintf

(* --- the graph: x f32, ids i64, keep bool (all [2;3]);
       y = x + float(ids), z = float(keep) * x --- *)

let tensor_meta dtype sizes =
  jstr
    {|{"dtype":%d,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    dtype
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

let as_tensor n = jstr {|{"as_tensor":{"name":"%s"}}|} n

let to_copy ~src ~dst =
  jstr
    {|{"target":"torch.ops.aten._to_copy.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dtype","arg":{"as_scalar_type":7},"kind":2},{"name":"non_blocking","arg":{"as_bool":false},"kind":2}],"outputs":[%s],"metadata":{}}|}
    (as_tensor src) (as_tensor dst)

let binary target ~a ~b ~dst =
  jstr
    {|{"target":"torch.ops.aten.%s","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"other","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    target (as_tensor a) (as_tensor b) (as_tensor dst)

let program ?(op = "add.Tensor") () =
  let inputs = [ ("x", 7); ("ids", 5); ("keep", 12) ] in
  let values = [ ("idsf", 7); ("keepf", 7); ("y", 7); ("z", 7) ] in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":false},"signature":{"input_specs":[%s],"output_specs":[%s]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (String.concat "," (List.map (fun (n, _) -> as_tensor n) inputs))
    (String.concat "," (List.map as_tensor [ "y"; "z" ]))
    (String.concat ","
       [
         to_copy ~src:"ids" ~dst:"idsf";
         to_copy ~src:"keep" ~dst:"keepf";
         binary op ~a:"x" ~b:"idsf" ~dst:"y";
         binary "mul.Tensor" ~a:"keepf" ~b:"x" ~dst:"z";
       ])
    (String.concat ","
       (List.map
          (fun (n, d) -> jstr {|"%s":%s|} n (tensor_meta d [ 2; 3 ]))
          (inputs @ values)))
    (String.concat ","
       (List.map
          (fun (n, _) -> jstr {|{"user_input":{"arg":%s}}|} (as_tensor n))
          inputs))
    (String.concat ","
       (List.map
          (fun o -> jstr {|{"user_output":{"arg":%s}}|} (as_tensor o))
          [ "y"; "z" ]))

(* --- torch.save-shaped tensor maps --- *)

type kind = Bool | Float | Long
type tensor = { name : string; kind : kind; sizes : int list; raw : string }

let f32s vs =
  String.concat ""
    (List.map
       (fun v ->
         let b = Bytes.create 4 in
         Bytes.set_int32_le b 0 (Int32.bits_of_float v);
         Bytes.to_string b)
       vs)

let i64s vs =
  String.concat ""
    (List.map
       (fun v ->
         let b = Bytes.create 8 in
         Bytes.set_int64_le b 0 (Int64.of_int v);
         Bytes.to_string b)
       vs)

let bools vs =
  String.concat "" (List.map (fun v -> if v then "\001" else "\000") vs)

let storage_class = function
  | Bool -> "BoolStorage"
  | Float -> "FloatStorage"
  | Long -> "LongStorage"

let torch_name = function
  | Bool -> "torch.bool"
  | Float -> "torch.float32"
  | Long -> "torch.int64"

let dtype_of = function
  | Bool -> Pt2_checkpoint_map.Dtype.BOOL
  | Float -> Pt2_checkpoint_map.Dtype.F32
  | Long -> Pt2_checkpoint_map.Dtype.I64

let le32 b v =
  Buffer.add_char b (Char.chr (v land 0xff));
  Buffer.add_char b (Char.chr ((v lsr 8) land 0xff));
  Buffer.add_char b (Char.chr ((v lsr 16) land 0xff));
  Buffer.add_char b (Char.chr ((v lsr 24) land 0xff))

let pickle (tensors : tensor list) =
  let b = Buffer.create 256 in
  let op c = Buffer.add_char b (Char.chr c) in
  let global m n =
    op 0x63;
    Buffer.add_string b (m ^ "\n" ^ n ^ "\n")
  in
  let unicode s =
    op 0x58;
    le32 b (String.length s);
    Buffer.add_string b s
  in
  let int1 v =
    op 0x4b;
    Buffer.add_char b (Char.chr v)
  in
  let tuple ints =
    op 0x28;
    List.iter int1 ints;
    op 0x74
  in
  let strides sizes =
    let rec go = function
      | [] -> []
      | _ :: rest as l -> List.fold_left ( * ) 1 (List.tl l) :: go rest
    in
    go sizes
  in
  op 0x80;
  Buffer.add_char b '\002';
  op 0x7d (* EMPTY_DICT *);
  op 0x28 (* MARK *);
  List.iteri
    (fun i t ->
      unicode t.name;
      global "torch._utils" "_rebuild_tensor_v2";
      op 0x28;
      op 0x28;
      unicode "storage";
      global "torch" (storage_class t.kind);
      unicode (string_of_int i);
      unicode "cpu";
      int1 (List.fold_left ( * ) 1 t.sizes);
      op 0x74;
      op 0x51;
      int1 0;
      tuple t.sizes;
      tuple (strides t.sizes);
      op 0x89;
      op 0x7d;
      op 0x74;
      op 0x52)
    tensors;
  op 0x75 (* SETITEMS *);
  op 0x2e;
  Buffer.contents b

let make_zip entries =
  let add z (path, data) =
    let file =
      match Zipc.File.stored_of_binary_string data with
      | Ok f -> f
      | Error e -> failwith e
    in
    match Zipc.Member.make ~path (Zipc.Member.File file) with
    | Ok m -> Zipc.add m z
    | Error e -> failwith e
  in
  match Zipc.to_binary_string (List.fold_left add Zipc.empty entries) with
  | Ok s -> s
  | Error e -> failwith e

let pt_file tensors =
  make_zip
    (("archive/data.pkl", pickle tensors)
    :: List.mapi
         (fun i t -> (Printf.sprintf "archive/data/%d" i, t.raw))
         tensors)

(* The producer's content digest of [tensors], in the order given. *)
let content_digest tensors =
  let named =
    List.map
      (fun t ->
        let logical =
          match
            Err.payload
              (Pt2_fixture.Logical.of_bytes ~dtype:(dtype_of t.kind)
                 ~shape:(List.map Int64.of_int t.sizes)
                 t.raw)
          with
          | Ok l -> l
          | Error _ -> failwith "logical"
        in
        (t.name, logical))
      tensors
  in
  match Pt2_fixture.Tensor_digest.digest named with
  | Ok d -> Pt2_sha256.Digest.to_hex d
  | Error _ -> failwith "digest"

(* --- cases --- *)

let tensor name kind raw = { name; kind; sizes = [ 2; 3 ]; raw }

let case0_inputs =
  [
    tensor "x" Float (f32s [ 1.; 2.; 3.; 4.; 5.; 6. ]);
    tensor "ids" Long (i64s [ 10; 20; 30; 40; 50; 60 ]);
    tensor "keep" Bool (bools [ true; false; true; false; true; false ]);
  ]

let case0_outputs =
  [
    tensor "y" Float (f32s [ 11.; 22.; 33.; 44.; 55.; 66. ]);
    tensor "z" Float (f32s [ 1.; 0.; 3.; 0.; 5.; 0. ]);
  ]

let case1_inputs =
  [
    tensor "x" Float (f32s [ 0.5; -1.5; 2.; 0.; -0.; 8. ]);
    tensor "ids" Long (i64s [ 1; 2; 3; 4; 5; 6 ]);
    tensor "keep" Bool (bools [ false; false; false; true; true; true ]);
  ]

let case1_outputs =
  [
    tensor "y" Float (f32s [ 1.5; 0.5; 5.; 4.; 5.; 14. ]);
    tensor "z" Float (f32s [ 0.; -0.; 0.; 0.; -0.; 8. ]);
  ]

(* Python hashlib's digests of the same four tensor maps. *)
let case0_inputs_sha =
  "30cc7caf805b72f810b022e85c5d28af11fe8c030919c17764c07f845d7e7b21"

let case0_outputs_sha =
  "c1217f299e71ec620ef63374bd5e776b1eab0d3f405be6f0a44ed3f4689f26be"

let case1_inputs_sha =
  "f1405c18228227e0fdc1d271ce16989f6f33509b0ceccb36ce316dc20fc46e0d"

let case1_outputs_sha =
  "00720d45250bc9fb4c5f5ebb5c70d40b882081f2a42317acae6218f1f010615f"

type case = {
  inputs : tensor list;  (** What inputs.pt holds, in file order. *)
  outputs : tensor list;
  inputs_sha : string;  (** What the descriptor claims. *)
  outputs_sha : string;
}

let base_cases =
  [
    {
      inputs = case0_inputs;
      outputs = case0_outputs;
      inputs_sha = case0_inputs_sha;
      outputs_sha = case0_outputs_sha;
    };
    {
      inputs = case1_inputs;
      outputs = case1_outputs;
      inputs_sha = case1_inputs_sha;
      outputs_sha = case1_outputs_sha;
    };
  ]

(* --- the release around them --- *)

let artifact_id =
  "replay/task/reference/forward/fp32/dynamo/static/ckpt-bbbbbbbbbbbb"

let source_name = "src.safetensors"
let source_bytes = "\002\000\000\000\000\000\000\000{}"

let contract_json ?(atol = "1e-05") ?(rtol = "0.0001")
    ?(kwargs = {|["x","ids","keep"]|}) program =
  let spec (n, d, s) = jstr {|{"dtype":%S,"name":%S,"shape":[%s]}|} d n s in
  jstr
    {|{"artifact_id":%S,"call":{"args":[],"kwargs":%s,"outputs":"tensor_tuple"},"dialect":"functional","dynamic_constraints":{},"graph_sha256":%S,"inputs":[%s],"mutations":[],"outputs":[%s],"tolerances":{"atol":%s,"rtol":%s},"verified_cases":2,"schema_version":1}|}
    artifact_id kwargs (W.hex program)
    (String.concat ","
       (List.map spec
          [
            ("x", "float32", "2,3");
            ("ids", "int64", "2,3");
            ("keep", "bool", "2,3");
          ]))
    (String.concat ","
       (List.map spec [ ("y", "float32", "2,3"); ("z", "float32", "2,3") ]))
    atol rtol

let cases_json ?(atol = "1e-05") ?(rtol = "0.0001") cases =
  jstr
    {|{"artifact_id":%S,"cases":[%s],"schema_version":1,"tolerances":{"atol":%s,"rtol":%s}}|}
    artifact_id
    (String.concat ","
       (List.mapi
          (fun i c ->
            jstr
              {|{"id":"case-%02d","inputs":["x","ids","keep"],"inputs_sha256":%S,"outputs":["y","z"],"outputs_sha256":%S}|}
              i c.inputs_sha c.outputs_sha)
          cases))
    atol rtol

let empty_config = {|{"config":{}}|}

let captures_json program =
  jstr
    {|{"artifact_id":%S,"captures":[],"counts":{},"graph_sha256":%S,"module_prefixes":{},"schema_version":1}|}
    artifact_id (W.hex program)

let map_json program =
  jstr
    {|{"schema_version":2,"artifact_id":%S,"graph_sha256":%S,"model_id":"replay","sources":{"checkpoint":{"files":[%s]}},"tensors":{},"unmapped":[]}|}
    artifact_id (W.hex program)
    (jstr
       {|{"name":%S,"sha256":%S,"size":%d,"url":%S,"repo_id":"o/r","revision":"%s"}|}
       source_name (W.hex source_bytes)
       (String.length source_bytes)
       (W.url source_name) (String.make 40 'a'))

type t = { cohort : Pt2_fixture.Cohort.t; served : (string * string) list }

let build ?(program = program ()) ?contract ?cases_text ?(cases = base_cases) ()
    =
  let contract =
    match contract with Some c -> c | None -> contract_json program
  in
  let cases_text =
    match cases_text with Some c -> c | None -> cases_json cases
  in
  let map = map_json program in
  let members =
    [
      ("captures.json", captures_json program);
      ("cases.json", cases_text);
      ("contract.json", contract);
      ("data/constants/model_constants_config.json", empty_config);
      ("data/weights/model_weights_config.json", empty_config);
      ("models/model.json", program);
      ("models/safetensors.v2.json", map);
    ]
    @ List.concat
        (List.mapi
           (fun i c ->
             [
               (Printf.sprintf "cases/case-%02d/inputs.pt" i, pt_file c.inputs);
               (Printf.sprintf "cases/case-%02d/outputs.pt" i, pt_file c.outputs);
             ])
           cases)
  in
  let members = List.sort (fun (a, _) (b, _) -> String.compare a b) members in
  let archive = W.gzip (W.tar members) in
  let case_ids =
    String.concat ","
      (List.mapi (fun i _ -> Printf.sprintf {|"case-%02d"|} i) cases)
  in
  let manifest =
    jstr
      {|{"archive":{"name":"archive.tar.gz","sha256":%S,"size":%d},"artifact_id":%S,"cases":[%s],"contract_sha256":%S,"graph_sha256":%S,"map_v2":{"assets":[],"member":"models/safetensors.v2.json"},"members":{%s},"payload":null,"producer_commit":"p","schema_version":1}|}
      (W.hex archive) (String.length archive) artifact_id case_ids
      (W.hex contract) (W.hex program)
      (String.concat "," (List.map W.member_json members))
  in
  let archive_pin = W.pin_json ~name:"archive.tar.gz" ~data:archive in
  let manifest_pin = W.pin_json ~name:"manifest.json" ~data:manifest in
  let publication =
    jstr
      {|{"artifacts":[{"artifact_id":%S,"assets":{"archive":%s,"manifest":%s},"graph_sha256":%S}],"release_tag":"tag-1","repository":"o/r","schema_version":1}|}
      artifact_id archive_pin manifest_pin (W.hex program)
  in
  let cohort_text =
    jstr
      {|{"artifacts":[{"archive":%s,"artifact_id":%S,"cases":[%s],"contract_sha256":%S,"graph_sha256":%S,"manifest":%s,"map_member":"models/safetensors.v2.json","map_sha256":%S,"map_sources":{"checkpoint":{"files":[%s]}},"role":"test"}],"publication":{"sha256":%S,"size":%d,"url":%S},"release_producer_commit":"p","release_tag":"tag-1","repository":"o/r","schema_version":1}|}
      archive_pin artifact_id case_ids (W.hex contract) (W.hex program)
      manifest_pin (W.hex map)
      (W.pin_json ~name:source_name ~data:source_bytes)
      (W.hex publication)
      (String.length publication)
      (W.url "publication.json")
  in
  {
    cohort = W.decode_cohort cohort_text;
    served =
      [
        (W.url "publication.json", publication);
        (W.url "manifest.json", manifest);
        (W.url "archive.tar.gz", archive);
        (W.url source_name, source_bytes);
      ];
  }
