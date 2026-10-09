(* A synthetic bundle for a six-capture graph.

     w  PARAMETER        F32 [2;3]  checkpoint, uncast       (floats 0..5)
     h  PARAMETER        F32 [2]    checkpoint, BF16 -> F32  (1.5, -2.0)
     b  BUFFER           I64 [3]    generated fill           (int64 1)
     c  CONSTANT_TENSOR  F32 []     inline                   (2.5)
     e  BUFFER           F32 [0]    generated empty
     k  PARAMETER        F32 [2]    pack                     (7.0, 8.0)

   Every digest below was produced by Python's hashlib over the little-endian
   bytes in the comment, not by the library under test. *)

module Map = Pt2_checkpoint_map

let jstr = Printf.sprintf

let artifact_id =
  "toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa"

let sha_w = "e2c0a71510b5394df7773b63fb5f54372b84c3564e67811bde7d665be227976d"
let sha_h = "252b3318179cc24998f3670913d52d39085cf65b0dfa98fa523ffeab4b6683fe"
let sha_b = "605390e5a369ee568b19ead1733af824c7c1d286d7d24b86283238fc44a99334"
let sha_c = "072e3304b03423a4767d28c5fed09f81d5190ff60a3d078c6c1350eeb8bee28b"
let sha_e = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
let sha_k = "621e6b9d912e2d0c9b2fc35bfd56ec75345026a6e61d0821763646eabacf47aa"

let program_json =
  {|{"graph_module":{"graph":{"inputs":[{"as_tensor":{"name":"w"}},{"as_tensor":{"name":"h"}},{"as_tensor":{"name":"b"}},{"as_tensor":{"name":"c"}},{"as_tensor":{"name":"e"}},{"as_tensor":{"name":"k"}},{"as_tensor":{"name":"x"}}],"outputs":[{"as_tensor":{"name":"x"}}],"nodes":[],"tensor_values":{},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[{"parameter":{"arg":{"name":"w"},"parameter_name":"w"}},{"parameter":{"arg":{"name":"h"},"parameter_name":"h"}},{"buffer":{"arg":{"name":"b"},"buffer_name":"b","persistent":false}},{"tensor_constant":{"arg":{"name":"c"},"tensor_constant_name":"c"}},{"buffer":{"arg":{"name":"e"},"buffer_name":"e","persistent":false}},{"parameter":{"arg":{"name":"k"},"parameter_name":"k"}},{"user_input":{"arg":{"as_tensor":{"name":"x"}}}}],"output_specs":[{"user_output":{"arg":{"as_tensor":{"name":"x"}}}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}

let ints l = String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) l)

let entry ?strides ~dtype ~sizes ~param name =
  let strides =
    match strides with
    | Some s -> s
    | None ->
        let rec go = function
          | [] -> []
          | _ :: rest as l -> List.fold_left ( * ) 1 (List.tl l) :: go rest
        in
        go sizes
  in
  jstr
    {|%S:{"path_name":"p_%s","is_param":%b,"use_pickle":false,"tensor_meta":{"dtype":%d,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu","index":null},"strides":[%s],"storage_offset":{"as_int":0},"layout":7}}|}
    name name param dtype (ints sizes) (ints strides)

(* torch ScalarType codes: 5 = LONG, 7 = FLOAT. *)
let weights_json ?(w_strides = [ 3; 1 ]) ?(h_sizes = [ 2 ]) () =
  "{\"config\":{"
  ^ String.concat ","
      [
        entry ~dtype:7 ~sizes:[ 2; 3 ] ~strides:w_strides ~param:true "w";
        entry ~dtype:7 ~sizes:h_sizes ~param:true "h";
        entry ~dtype:7 ~sizes:[ 2 ] ~param:true "k";
      ]
  ^ "}}"

let constants_json =
  "{\"config\":{"
  ^ String.concat ","
      [
        entry ~dtype:5 ~sizes:[ 3 ] ~param:false "b";
        entry ~dtype:7 ~sizes:[] ~param:false "c";
        entry ~dtype:7 ~sizes:[ 0 ] ~param:false "e";
      ]
  ^ "}}"

let capture ~target ~kind ~dtype ~shape ~sha =
  jstr
    {|{"dtype":%S,"kind":%S,"live":true,"payload_name":"p_%s","referenced":true,"scalar":%b,"shape":[%s],"source":{"kind":"payload","value_sha256":%S},"ssa_name":"s_%s","target":%S}|}
    dtype kind target (shape = "") shape sha target target

let captures_json ?(graph_sha256 = fun s -> s) ?(artifact_id = artifact_id)
    graph =
  jstr
    {|{"artifact_id":%S,"captures":[%s],"counts":{"BUFFER":2,"CONSTANT_TENSOR":1,"PARAMETER":3},"graph_sha256":%S,"module_prefixes":{},"schema_version":1}|}
    artifact_id
    (String.concat ","
       [
         capture ~target:"w" ~kind:"PARAMETER" ~dtype:"float32" ~shape:"2,3"
           ~sha:sha_w;
         capture ~target:"h" ~kind:"PARAMETER" ~dtype:"float32" ~shape:"2"
           ~sha:sha_h;
         capture ~target:"b" ~kind:"BUFFER" ~dtype:"int64" ~shape:"3" ~sha:sha_b;
         capture ~target:"c" ~kind:"CONSTANT_TENSOR" ~dtype:"float32" ~shape:""
           ~sha:sha_c;
         capture ~target:"e" ~kind:"BUFFER" ~dtype:"float32" ~shape:"0"
           ~sha:sha_e;
         capture ~target:"k" ~kind:"PARAMETER" ~dtype:"float32" ~shape:"2"
           ~sha:sha_k;
       ])
    (graph_sha256 (Pt2_sha256.Digest.to_hex (Pt2_sha256.string graph)))

(* --- sources: real safetensors bytes, so the pins below are the files' --- *)

let le32 f =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.bits_of_float f);
  Bytes.to_string b

let le64 i =
  let b = Bytes.create 8 in
  Bytes.set_int64_le b 0 i;
  Bytes.to_string b

(* A safetensors file: header length, JSON header, then the tensors back to
   back in the order given. *)
let safetensors items =
  let _, parts =
    List.fold_left
      (fun (off, acc) (name, dtype, shape, raw) ->
        let len = String.length raw in
        ( off + len,
          jstr {|%S:{"dtype":%S,"shape":[%s],"data_offsets":[%d,%d]}|} name
            dtype
            (String.concat "," (List.map string_of_int shape))
            off (off + len)
          :: acc ))
      (0, []) items
  in
  let header = "{" ^ String.concat "," (List.rev parts) ^ "}" in
  let prefix = Bytes.create 8 in
  Bytes.set_int64_le prefix 0 (Int64.of_int (String.length header));
  Bytes.to_string prefix ^ header
  ^ String.concat "" (List.map (fun (_, _, _, r) -> r) items)

let w_bytes = String.concat "" (List.map le32 [ 0.; 1.; 2.; 3.; 4.; 5. ])
let h_stored = "\xc0\x3f\x00\xc0" (* BF16 1.5, -2.0 *)
let k_bytes = String.concat "" (List.map le32 [ 7.; 8. ])

let toy_bytes =
  safetensors
    [
      ("model.w", "F32", [ 2; 3 ], w_bytes);
      ("model.h", "BF16", [ 2 ], h_stored);
      ( "model.other",
        "I64",
        [ 3 ],
        String.concat "" [ le64 5L; le64 6L; le64 7L ] );
    ]

let pack_bytes = safetensors [ ("k", "F32", [ 2 ], k_bytes) ]
let hex_of s = Pt2_sha256.Digest.to_hex (Pt2_sha256.string s)

let file_pin ?(sha = "") ?(size = -1) ~name ~data ~extra () =
  jstr {|{"name":%S,"sha256":%S,"size":%d,"url":"https://example.org/%s"%s}|}
    name
    (if sha = "" then hex_of data else sha)
    (if size < 0 then String.length data else size)
    name extra

let rev = String.make 40 'a'

let toy_pin ?sha ?size () =
  file_pin ?sha ?size ~name:"toy.safetensors" ~data:toy_bytes
    ~extra:(jstr {|,"repo_id":"o/toy","revision":"%s"|} rev)
    ()

let pack_pin () =
  file_pin ~name:"pack.safetensors" ~data:pack_bytes ~extra:"" ()

let map_json =
  jstr
    {|{"schema_version":2,"artifact_id":%S,"graph_sha256":%S,"model_id":"toy","sources":{"checkpoint":{"files":[%s]},"graph_owned":%s},"tensors":{%s},"unmapped":[]}|}
    artifact_id
    (Pt2_sha256.Digest.to_hex (Pt2_sha256.string program_json))
    (toy_pin ()) (pack_pin ())
    (String.concat ","
       [
         jstr
           {|"w":{"dtype":"F32","shape":[2,3],"sha256":%S,"origin":{"kind":"checkpoint","file":"toy.safetensors","key":"model.w","convert":{"op":"none"},"tied_aliases":[]}}|}
           sha_w;
         jstr
           {|"h":{"dtype":"F32","shape":[2],"sha256":%S,"origin":{"kind":"checkpoint","file":"toy.safetensors","key":"model.h","convert":{"op":"cast","from":"BF16","to":"F32"},"tied_aliases":["model.h2"]}}|}
           sha_h;
         jstr
           {|"b":{"dtype":"I64","shape":[3],"sha256":%S,"origin":{"kind":"generated","op":"fill","element_hex":"0100000000000000"}}|}
           sha_b;
         jstr
           {|"c":{"dtype":"F32","shape":[],"sha256":%S,"origin":{"kind":"inline","data_base64":"AAAgQA=="}}|}
           sha_c;
         jstr
           {|"e":{"dtype":"F32","shape":[0],"sha256":%S,"origin":{"kind":"generated","op":"empty"}}|}
           sha_e;
         jstr
           {|"k":{"dtype":"F32","shape":[2],"sha256":%S,"origin":{"kind":"pack","key":"k"}}|}
           sha_k;
       ])

(* A substring replacement that fails the test if it matched nothing, so a
   corruption can never silently be the identity. *)
let replace ~sub ~by s =
  let n = String.length sub in
  let rec find i =
    if i + n > String.length s then failwith ("fixture lacks " ^ sub)
    else if String.sub s i n = sub then i
    else find (i + 1)
  in
  let i = find 0 in
  String.sub s 0 i ^ by ^ String.sub s (i + n) (String.length s - i - n)

let decode_program () =
  match Err.payload (Pt2_archive.program_of_json program_json) with
  | Ok p -> p
  | Error _ -> failwith "program fixture"

let decode_config s =
  match Err.payload (Pt2_archive.weights_config_of_json s) with
  | Ok c -> c
  | Error _ -> failwith "config fixture"

let graph ?(program = program_json) ?(weights = weights_json ())
    ?(constants = constants_json) ?captures ?(artifact_id = artifact_id) () :
    Map.Validate.graph =
  let captures =
    match captures with Some c -> c | None -> captures_json program_json
  in
  let captures =
    match Err.payload (Map.Captures.of_string captures) with
    | Ok c -> c
    | Error _ -> failwith "captures fixture"
  in
  let decoded =
    match Err.payload (Pt2_archive.program_of_json program) with
    | Ok p -> p
    | Error _ -> failwith "program fixture"
  in
  {
    Map.Validate.artifact_id;
    captures;
    constants = decode_config constants;
    graph_digest = Pt2_sha256.string program;
    program = decoded;
    weights = decode_config weights;
  }
