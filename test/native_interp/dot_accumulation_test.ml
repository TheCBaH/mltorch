(* How Direct accumulates a dot product, pinned against two independent
   oracles. The engine computes in binary64 and rounds once when a value is
   stored as binary32 (see the numerics contract in the design record), so a
   linear layer's output is the CORRECTLY ROUNDED exact dot product. A reference
   that accumulates sequentially in binary32 -- as the Transformers fixtures'
   torch CPU references do -- differs from it by accumulated rounding noise that
   grows with the reduction length. This test says both things, so that a later
   change of accumulation order or precision in Direct cannot slip in as an
   optimization: it would change every golden, and it would move the engine
   away from the exact answer. *)

let jstr = Printf.sprintf

let tensor_meta sizes =
  jstr
    {|{"dtype":7,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

let as_tensor n = jstr {|{"as_tensor":{"name":"%s"}}|} n

let program ~k ~n =
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s,%s],"outputs":[%s],"nodes":[{"target":"torch.ops.aten.linear.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}],"tensor_values":{"x":%s,"w":%s,"y":%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[{"user_input":{"arg":%s}},{"user_input":{"arg":%s}}],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (as_tensor "x") (as_tensor "w") (as_tensor "y") (as_tensor "x")
    (as_tensor "w") (as_tensor "y")
    (tensor_meta [ 1; k ])
    (tensor_meta [ n; k ])
    (tensor_meta [ 1; n ])
    (as_tensor "x") (as_tensor "w") (as_tensor "y")

let r32 v = Int32.float_of_bits (Int32.bits_of_float v)

(* A deterministic stream of binary32 values in [-1, 1) with mixed magnitudes,
   so that partial sums grow and lose low bits. *)
let values count ~seed =
  let state = ref seed in
  Array.init count (fun _ ->
      state := ((!state * 1103515245) + 12345) land 0x7fffffff;
      let u = float_of_int (!state lsr 8) /. float_of_int (1 lsl 23) in
      let magnitude = if !state land 3 = 0 then 1.0 else 0.03 in
      r32 (((2. *. u) -. 1.) *. magnitude))

let pt2_tensor sizes data =
  let b = Bytes.create (4 * Array.length data) in
  Array.iteri
    (fun i v -> Bytes.set_int32_le b (4 * i) (Int32.bits_of_float v))
    data;
  {
    Pt2_tensor.dtype = Pt2_dtype.Float32;
    sizes;
    strides = (match sizes with [ _; k ] -> [ k; 1 ] | _ -> [ 1 ]);
    storage_offset = 0;
    data = Pt2_storage.of_string (Bytes.to_string b);
  }

(* The exact dot product, correctly rounded to binary32. Each product of two
   binary32 values is exact in binary64; the running sum is kept as an
   unevaluated pair with every addition's error carried (TwoSum), so the total
   is exact to well beyond binary32. *)
let exact_rounded x w =
  let hi = ref 0. and lo = ref 0. in
  Array.iteri
    (fun j xj ->
      let p = xj *. w.(j) in
      let s = !hi +. p in
      let bv = s -. !hi in
      let e = !hi -. (s -. bv) +. (p -. bv) in
      hi := s;
      lo := !lo +. e)
    x;
  r32 (!hi +. !lo)

(* What a binary32 sequential fused-multiply-add chain gives. *)
let sequential_fma x w =
  let acc = ref 0. in
  Array.iteri (fun j xj -> acc := r32 (Float.fma xj w.(j) !acc)) x;
  !acc

let%expect_test
    "Direct's linear is the correctly rounded dot; a binary32 chain is not" =
  let k = 12288 and n = 6 in
  let x = values k ~seed:7 in
  let w = Array.init n (fun i -> values k ~seed:(100 + i)) in
  let wflat = Array.concat (Array.to_list w) in
  let json = program ~k ~n in
  let archive =
    match
      Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont json
    with
    | Error e -> failwith e
    | Ok program ->
        let none =
          {
            Pytorch_weights_config.ModelWeightsConfig.config =
              Schema_runtime.String_map.empty;
          }
        in
        Pt2_archive.of_parts ~program ~weights:none ~constants:none
          ~load:(fun _ -> Error "no payload")
  in
  match
    Err.payload
      (Native_interp.run_named archive
         ~inputs:
           [ ("x", pt2_tensor [ 1; k ] x); ("w", pt2_tensor [ n; k ] wflat) ])
  with
  | Error e -> Fmt.pr "%a@." Native_interp.pp_error e
  | Ok [ out ] ->
      let native c = Tensor.read out (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c) in
      let exact_matches = ref 0 and chain_matches = ref 0 in
      let worst = ref 0. in
      for c = 0 to n - 1 do
        let got = native c in
        let exact = exact_rounded x w.(c) and chain = sequential_fma x w.(c) in
        if got = exact then incr exact_matches;
        if got = chain then incr chain_matches;
        worst := Float.max !worst (Float.abs (chain -. exact))
      done;
      Fmt.pr "native equals the correctly rounded dot: %d of %d@."
        !exact_matches n;
      Fmt.pr "native equals the sequential binary32 chain: %s@."
        (if !chain_matches = n then "all (the chain is no worse here)"
         else Printf.sprintf "%d of %d" !chain_matches n);
      Fmt.pr "the chain strays from the exact value: %b@." (!worst > 0.);
      [%expect
        {|
        native equals the correctly rounded dot: 6 of 6
        native equals the sequential binary32 chain: 0 of 6
        the chain strays from the exact value: true |}]
  | Ok _ -> print_endline "wrong output count"
