(* Section 3: the tensor payload JSON path -- Bigarray storage in, JSON text out,
   and back. Not melange-reachable: it needs lib/native (Bigarray) and
   Jsont_bytesrw.

   The bit-exactness verdict is printed SEPARATELY from the JSON text on
   purpose. If only the text differed we would know the divergence is in decimal
   formatting; if only the round-trip flag flipped we would know the storage or
   decode drifted. Collapsing them into one line would throw that away exactly
   when it is most useful. *)

let run () =
  print_endline "=== tensor-json ===";
  let values = Array.of_list Probe_walk_core.awkward in
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:(Array.length values) in
  let tensor =
    Tensor.materialize shape (fun c ->
        Walk_core.Float32.to_f32 values.((Vec6.offset shape c :> int)))
  in
  (* Failures RAISE rather than print. This is a differential harness: both
     sides run this same source, so a failure that reproduces on both -- which a
     genuine encode/decode bug would -- prints identical text, diffs clean, and
     passes. Only a non-zero exit gets noticed. Plain [result] from Jsont, so
     [failwith] rather than [Core.or_raise] (see [[testing_strategy]]). *)
  let json =
    match Graph_json.encode_tensor tensor with
    | Ok json -> json
    | Error e -> failwith ("probe: encode_tensor: " ^ e)
  in
  Printf.printf "json %s\n" json;
  let back =
    match Graph_json.decode_tensor json with
    | Ok back -> back
    | Error e -> failwith ("probe: decode_tensor: " ^ e)
  in
  let bit_exact = Tensor.equal_bits tensor back in
  Printf.printf "round-trip bit-exact: %b\n" bit_exact;
  if not bit_exact then failwith "probe: tensor round-trip is not bit-exact"
