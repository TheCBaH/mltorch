module M = Inference_compare_coverage.Manifest
module P = Inference_compare_pt2

let () = Inference_compare_providers.Providers.register_all ()
let views = [ "aten"; "native"; "native.canonical"; "native4d" ]

let configurations =
  [
    "sequential32";
    "release";
    "arena";
    "separate";
    "shared";
    "scheduled";
    "symbolic";
  ]

(* Every word a variant can carry: the views, the configuration words and every
   registered engine. A word added to the vocabulary has to be added here by
   whoever registers it, and then to a row. *)
let vocabulary () = views @ configurations @ P.Provider.names ()

let is_arena_bytes w =
  String.length w > 6
  && String.sub w 0 6 = "arena="
  && String.for_all
       (fun c -> c >= '0' && c <= '9')
       (String.sub w 6 (String.length w - 6))

let uncovered rows =
  let used = List.concat_map M.words rows in
  List.filter (fun w -> not (List.mem w used)) (vocabulary ())
  @ if List.exists is_arena_bytes used then [] else [ "arena=BYTES" ]

let unknown rows =
  let known = vocabulary () in
  List.sort_uniq compare
    (List.filter
       (fun w -> not (List.mem w known || is_arena_bytes w))
       (List.concat_map M.words rows))

let manifest () =
  let text =
    In_channel.with_open_bin "coverage.manifest" In_channel.input_all
  in
  match M.parse text with Ok rows -> rows | Error m -> failwith m

let%expect_test "the manifest covers the vocabulary and names no other word" =
  let rows = manifest () in
  Printf.printf "uncovered: [%s]\nunknown: [%s]\n"
    (String.concat "; " (uncovered rows))
    (String.concat "; " (unknown rows));
  [%expect {|
    uncovered: []
    unknown: [] |}]

let%expect_test "every row's variants parse" =
  List.iter
    (fun (r : M.row) ->
      List.iter
        (fun v ->
          match P.Variant.of_string v with
          | Ok _ -> ()
          | Error m ->
              (* a row that expects `invalid` may name an impossible variant *)
              if r.expect <> M.Invalid then Printf.printf "%s: %s\n" r.id m)
        [ r.left; r.right ])
    (manifest ());
  [%expect {| |}]

(* The check can fail: a manifest without a row for a word is reported. *)
let%expect_test "a missing row is found" =
  let rows =
    match M.parse "a | native | native.canonical | - | agree | -\n" with
    | Ok r -> r
    | Error m -> failwith m
  in
  Printf.printf "%d words uncovered; rivet among them: %b\n"
    (List.length (uncovered rows))
    (List.mem "rivet" (uncovered rows));
  Printf.printf "unknown: [%s]\n"
    (String.concat "; "
       (unknown
          (match M.parse "b | native:fast | aten | - | agree | -\n" with
          | Ok r -> r
          | Error m -> failwith m)));
  [%expect
    {|
    22 words uncovered; rivet among them: true
    unknown: [fast] |}]

let%expect_test "a malformed manifest is rejected, naming the line" =
  List.iter
    (fun text ->
      match M.parse text with
      | Ok rows -> Printf.printf "%d rows\n" (List.length rows)
      | Error m -> print_endline m)
    [
      "# only a comment\n\n";
      "a | native | aten | - | agree\n";
      "a | native | aten | - | maybe | -\n";
      "a | native | aten | TOL | agree | -\n";
      "a | native | aten | - | agree | text\n";
      "a | native | aten | - | refused | -\n";
      "a | native | aten | - | agree | -\na | aten | native | - | agree | -\n";
      "a | native | aten | TOL=0 DONOR=right | refused no gcc | +x;-y\n";
    ];
  [%expect
    {|
    0 rows
    line 1: 5 fields, expected 6
    line 1: unknown expectation "maybe"
    line 1: environment "TOL" is not NAME=VALUE
    line 1: check "text" starts with + or -
    line 1: unknown expectation "refused"
    line 2: duplicate id a
    1 rows |}]
