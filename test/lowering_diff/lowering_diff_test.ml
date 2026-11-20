open Lowering_diff

(* Applying the script to the old lines must give the new lines, and its Equal
   and Delete lines must be the old ones, in order. *)
let check a b =
  let t = diff a b in
  let old_ =
    List.filter_map
      (function Equal l | Delete l -> Some l | Insert _ -> None)
      t.ops
  in
  let new_ =
    List.filter_map
      (function Equal l | Insert l -> Some l | Delete _ -> None)
      t.ops
  in
  old_ = lines a && new_ = lines b

let show a b =
  let t = diff a b in
  Fmt.pr "valid=%b minimal=%b identical=%b@." (check a b) t.minimal
    (identical t);
  print_string (unified ~context:1 ~old_label:"a" ~new_label:"b" t)

let%expect_test "identity" =
  show "x\ny\n" "x\ny\n";
  [%expect {| valid=true minimal=true identical=true |}]

let%expect_test "replace, insert, delete" =
  show "a\nb\nc\nd\ne\nf\ng\n" "a\nB\nc\nd\ne\nf\nG\nh\n";
  [%expect
    {|
    valid=true minimal=true identical=false
    --- a
    +++ b
    @@ -1,3 +1,3 @@
     a
    -b
    +B
     c
    @@ -6,2 +6,3 @@
     f
    -g
    +G
    +h
    |}]

let%expect_test "empty sides" =
  show "" "p\nq\n";
  show "p\nq\n" "";
  show "" "";
  [%expect
    {|
    valid=true minimal=true identical=false
    --- a
    +++ b
    @@ -0,0 +1,2 @@
    +p
    +q
    valid=true minimal=true identical=false
    --- a
    +++ b
    @@ -1,2 +0,0 @@
    -p
    -q
    valid=true minimal=true identical=true
    |}]

let%expect_test "trailing newline is not a line; CR stays" =
  Fmt.pr "%d %d %d@."
    (List.length (lines "a\nb"))
    (List.length (lines "a\nb\n"))
    (List.length (lines "a\r\nb\r\n"));
  Fmt.pr "%b@." (identical (diff "a\nb" "a\nb\n"));
  Fmt.pr "%b@." (identical (diff "a\r\n" "a\n"));
  [%expect {|
    2 2 2
    true
    false
    |}]

let%expect_test "past the search bound the script is still correct" =
  let big n tag =
    String.concat "\n" (List.init n (fun i -> Printf.sprintf "%s%d" tag i))
  in
  let a = big 2000 "x" and b = big 2000 "y" in
  let t = diff a b in
  Fmt.pr "valid=%b minimal=%b@." (check a b) t.minimal;
  [%expect {| valid=true minimal=false |}]

let%expect_test "many scattered edits stay valid" =
  let a = List.init 300 (fun i -> Printf.sprintf "l%d" i) in
  let b = List.mapi (fun i l -> if i mod 7 = 0 then l ^ "'" else l) a in
  let b = List.filteri (fun i _ -> i mod 11 <> 5) b in
  let a = String.concat "\n" a and b = String.concat "\n" b in
  Fmt.pr "valid=%b minimal=%b@." (check a b) (diff a b).minimal;
  [%expect {| valid=true minimal=true |}]
