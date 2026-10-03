(* A synthetic checkpoint and a one-parameter program, shared by the pure and
   unix tests. *)

(* One parameter [w], f32 [2; 3], row-major, in a three-tensor checkpoint:
   [W] is the match, [I] an int64 tensor of the same element count, [S] a
   differently shaped f32. *)

let jstr = Printf.sprintf

let program_json =
  {|{"graph_module":{"graph":{"inputs":[{"as_tensor":{"name":"w"}}],"outputs":[{"as_tensor":{"name":"w"}}],"nodes":[],"tensor_values":{},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[{"parameter":{"arg":{"name":"w"},"parameter_name":"w"}}],"output_specs":[{"user_output":{"arg":{"as_tensor":{"name":"w"}}}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}

let ints l = String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) l)

let weights ?(dtype = 7) ?(sizes = [ 2; 3 ]) ?(strides = [ 3; 1 ]) () =
  jstr
    {|{"config":{"w":{"path_name":"weight_0","is_param":true,"use_pickle":false,"tensor_meta":{"dtype":%d,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu","index":null},"strides":[%s],"storage_offset":{"as_int":0},"layout":7}}}}|}
    dtype (ints sizes) (ints strides)

let checkpoint =
  let entries =
    [
      ("W", "F32", [ 2; 3 ], 24);
      ("I", "I64", [ 3 ], 24);
      ("S", "F32", [ 4 ], 16);
    ]
  in
  let _, parts =
    List.fold_left
      (fun (off, acc) (name, dtype, shape, len) ->
        ( off + len,
          jstr {|%S:{"dtype":%S,"shape":[%s],"data_offsets":[%d,%d]}|} name
            dtype
            (String.concat "," (List.map string_of_int shape))
            off (off + len)
          :: acc ))
      (0, []) entries
  in
  let header = "{" ^ String.concat "," (List.rev parts) ^ "}" in
  let data =
    (* W holds the floats 0..5; the rest is filler. *)
    let b = Buffer.create 64 in
    for i = 0 to 5 do
      Buffer.add_int32_le b (Int32.bits_of_float (float_of_int i))
    done;
    Buffer.add_string b (String.make (24 + 16) '\x00');
    Buffer.contents b
  in
  let prefix = Bytes.create 8 in
  Bytes.set_int64_le prefix 0 (Int64.of_int (String.length header));
  Bytes.to_string prefix ^ header ^ data

let memory =
  match Safetensors.Memory.of_string checkpoint with
  | Ok m -> m
  | Error e -> failwith (Format.asprintf "%a" Safetensors.Error.pp e)

let map_json ?(version = 1) ?(unmapped = []) ?(key = "W") ?(dtype = "F32")
    ?(shape = [ 2; 3 ]) ?(sha256 = String.make 64 'b')
    ?(size = String.length checkpoint) () =
  jstr
    {|{"schema_version":%d,"source":{"filename":"model.safetensors","repo_id":"o/m","revision":"%s","sha256":"%s","size":%d,"url":"https://huggingface.co/o/m/resolve/r/model.safetensors"},"tensors":{"w":{"dtype":%S,"key":%S,"shape":[%s]}},"unmapped":[%s]}|}
    version (String.make 40 'a') sha256 size dtype key
    (String.concat "," (List.map string_of_int shape))
    (String.concat "," (List.map (jstr "%S") unmapped))
