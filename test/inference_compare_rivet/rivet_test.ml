module R = Inference_compare_rivet.Providers
module Body = Inference_compare_generated_test.Generated_body
module P = Inference_compare_pt2
module V = P.Variant

let%expect_test
    "every mechanism and allocator gives the direct evaluator's bits" =
  let run name (module E : P.Graph_runner.ENGINE with type op = Graph_ir.op) =
    print_endline name;
    let module C = Body.Check (E) in
    C.run ()
  in
  run "image" (module R.Image_default);
  run "image, scanned" (module R.Image_scanned);
  run "table" (module R.Table_default);
  run "table, scanned" (module R.Table_scanned);
  [%expect
    {|
    image
    seed 0: true
    seed 3: true
    image, scanned
    seed 0: true
    seed 3: true
    table
    seed 0: true
    seed 3: true
    table, scanned
    seed 0: true
    seed 3: true |}]

(* The check can fail: the same engine's results against the evaluator run on
   other inputs differ. *)
let%expect_test "a deviation is seen" =
  let g = Body.chain in
  match R.Table_default.compile g with
  | Error m -> print_endline m
  | Ok c ->
      (match R.Table_default.execute c g ~inputs:[ Body.input_of g 0 ] with
      | Error m -> print_endline m
      | Ok outs ->
          Printf.printf "other inputs: %b\n"
            (List.for_all2 Tensor.equal_bits outs
               (Body.direct g [ Body.input_of g 3 ])));
      R.Table_default.release c;
      [%expect {| other inputs: false |}]

let identity name =
  match P.Provider.instantiate name with
  | Ok i -> i.identity
  | Error m -> failwith m

let%expect_test "the words are variants and each has its own identity" =
  R.register ();
  List.iter
    (fun s ->
      match V.of_string s with
      | Ok v -> print_endline (V.to_string v)
      | Error m -> print_endline m)
    [
      "native:rivet";
      "native.canonical:rivet-table-scanned";
      "native4d:rivet";
      "native:rivet,arena";
    ];
  let words =
    [ "rivet"; "rivet-scanned"; "rivet-table"; "rivet-table-scanned" ]
  in
  List.iter
    (fun a ->
      List.iter
        (fun b ->
          if a < b then
            Printf.printf "%s = %s: %b\n" a b (identity a = identity b))
        words)
    words;
  Printf.printf "rivet = rivet: %b\n" (identity "rivet" = identity "rivet");
  [%expect
    {|
    native:rivet
    native.canonical:rivet-table-scanned
    variant "native4d:rivet": no configuration "rivet"
    variant "native:rivet,arena": rivet takes no other word
    rivet = rivet-scanned: false
    rivet = rivet-table: false
    rivet = rivet-table-scanned: false
    rivet-scanned = rivet-table: false
    rivet-scanned = rivet-table-scanned: false
    rivet-table = rivet-table-scanned: false
    rivet = rivet: true |}]
