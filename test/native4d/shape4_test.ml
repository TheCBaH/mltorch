(* [Axis4] and [Shape4]: that the dialect's invariant is a type property rather
   than a check someone has to remember to call. *)

open Native4d

let pp_shape_result =
  Core.Pretty.err_result ~ok:Shape4.pp ~error:(Shape4.pp_error :> _ Fmt.t)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c

(* Where a substring starts, if anywhere. Backtrace assertions are structural,
   never golden (test/native/core_test.ml does the same); this returns the
   offset rather than a bool so a test can compare frame ORDER. *)
let index_of s sub =
  let n = String.length s and m = String.length sub in
  let rec go i =
    if i + m > n then None
    else if String.sub s i m = sub then Some i
    else go (i + 1)
  in
  go 0

(* ---- Axis4 ---------------------------------------------------------------- *)

(* [of_axis] is partial, and that partiality is the design: a caller converting a
   Native axis has to say what it does with the two the dialect cannot name. *)
let%expect_test "axis4: T and D have no four-axis name" =
  List.iter
    (fun axis ->
      Format.printf "%a -> %a@." Axis.pp axis
        (Core.Pretty.option_or ~none:"outside the dialect" Axis4.pp)
        (Axis4.of_axis axis))
    Axis.all;
  [%expect
    {|
    N -> N
    T -> outside the dialect
    D -> outside the dialect
    H -> H
    W -> W
    C -> C |}]

(* [all] is in FRAME order, not the alphabetical order the constructors are
   declared in — every consumer iterating axes cares about the frame. *)
let%expect_test "axis4: all is in frame order" =
  Format.printf "%a@." (Fmt.list ~sep:(Fmt.any " ") Axis4.pp) Axis4.all;
  [%expect {| N H W C |}]

(* ---- Shape4 --------------------------------------------------------------- *)

let%expect_test "shape4: what of_vec6 accepts" =
  List.iter
    (fun (label, shape) ->
      Format.printf "%-22s %a@." label pp_shape_result (Shape4.of_vec6 shape))
    [
      ("four-axis", s 2 1 1 4 4 3);
      ("all unit", s 1 1 1 1 1 1);
      ("non-unit T", s 1 2 1 4 4 3);
      ("non-unit D", s 1 1 2 4 4 3);
      ("both", s 1 2 2 4 4 3);
    ];
  [%expect
    {|
    four-axis              [N=2 H=4 W=4 C=3]
    all unit               [N=1 H=1 W=1 C=1]
    non-unit T             shape has extent on T or D: [T=2 D=1 H=4 W=4 C=3]
    non-unit D             shape has extent on T or D: [D=2 H=4 W=4 C=3]
    both                   shape has extent on T or D: [T=2 D=2 H=4 W=4 C=3] |}]

(* [Shape4.pp] deliberately does not trim leading 1s the way [Vec6.pp_shape]
   does: a test reading the output can tell which type produced it. *)
let%expect_test "shape4: printing is distinguishable from Vec6" =
  let v = s 1 1 1 1 1 8 in
  let s4 = Shape4.of_vec6 v |> Result.get_ok in
  Format.printf "Vec6:   %a@." Vec6.pp_shape v;
  Format.printf "Shape4: %a@." Shape4.pp s4;
  [%expect {|
    Vec6:   [C=8]
    Shape4: [N=1 H=1 W=1 C=8] |}]

(* [set] is total, unlike a [Vec6.set] on T or D — [Axis4.t] cannot name those,
   so no sequence of sets can take a shape out of the dialect. *)
let%expect_test "shape4: set cannot leave the dialect" =
  let s4 = Shape4.of_ints ~n:1 ~h:4 ~w:4 ~c:3 in
  let moved =
    List.fold_left
      (fun acc axis -> Shape4.set acc axis (Dim.extent 7))
      s4 Axis4.all
  in
  Format.printf "%a -> %a@." Shape4.pp s4 Shape4.pp moved;
  Format.printf "still four-axis: %a@." Vec6.pp_shape (Shape4.to_vec6 moved);
  [%expect
    {|
    [N=1 H=4 W=4 C=3] -> [N=7 H=7 W=7 C=7]
    still four-axis: [N=7 T=1 D=1 H=7 W=7 C=7] |}]

let%expect_test "shape4: round-trips through JSON" =
  let s4 = Shape4.of_ints ~n:2 ~h:3 ~w:5 ~c:7 in
  let json = Json_util.enc Shape4.jsont s4 in
  let back = Json_util.dec Shape4.jsont json in
  Format.printf "equal: %b (%a)@." (Shape4.equal s4 back) Shape4.pp back;
  [%expect {| equal: true ([N=2 H=3 W=5 C=7]) |}]

(* A six-axis shape outside the dialect must not decode into one. Encoding
   cannot produce such a document, but a hand-written or older one can. *)
let%expect_test "shape4: decoding rejects a non-four-axis document" =
  let json = Json_util.enc Vec6.shape_jsont (s 1 1 9 4 4 3) in
  Format.printf "%s@."
    (match Json_util.dec Shape4.jsont json with
    | exception Jsont.Error _ -> "rejected"
    | s4 -> Format.asprintf "accepted %a" Shape4.pp s4);
  [%expect {| rejected |}]

(* ---- the dialect through the shared functors ------------------------------ *)

(* [Dialect4.validate_sig] is the hook without which the four-axis invariant
   leaks: shape inference constrains what OPS produce, and a graph input or a
   captured constant is produced by no op. So a Native4D graph whose input is
   directly its output would otherwise validate with extent on T or D.

   Built by hand rather than through [Builder], which cannot express it —
   [Shape4] guards construction, and that is the point: this is the hole a
   hand-assembled or deserialised graph could still fall into. *)
let%expect_test "view4: an input with a non-four-axis signature is rejected" =
  let id = Tensor_id.of_int 0 in
  let graph_with shape =
    {
      Graph_common.Graph.nodes = [];
      root =
        {
          Graph_common.Group.id = Graph_common.Group_id.of_int 0;
          label = None;
          items = [];
        };
      tensors =
        Tensor_id.Map.singleton id
          (Tensor_sig.create ~id ~name:"" ~shape ~fmt:(Payload.Fmt Payload.F32)
             ());
      inputs = [ id ];
      input_kinds = Tensor_id.Map.empty;
      outputs = [ id ];
    }
  in
  let check label shape =
    Format.printf "%-14s %s@." label
      (match Framework.View4.of_graph (graph_with shape) with
      | Ok _ -> "accepted"
      | Error e ->
          Format.asprintf "%a" Framework.View4.pp_error (Err.Error.kind e))
  in
  check "four-axis" (s 1 1 1 4 4 3);
  check "extent on D" (s 1 1 5 4 4 3);
  [%expect
    {|
    four-axis      accepted
    extent on D    tensor t0 is not a legal signature for this dialect: shape has extent on T or D:
                                                         [D=5 H=4 W=4 C=3] |}]

(* The SAME rejection, asked a different question: where was it detected?

   [`Invalid_sig] carries the dialect's payload out of [D.validate_sig], and the
   [Err.Error.t] wrapping it must be the one [Shape4.of_vec6] built — not a
   fresh one captured at the point the view re-raised. Rebuilding the error by
   hand (Err.fail applied to (Err.Error.kind e)) type-checks, prints the same
   message, and passes the test above, while silently moving the detection site
   to the caller. [Err.map_error] is what keeps it.

   Asserts a property, not backtrace text: frame content shifts per build, so
   this checks only which MODULE names the detection frame, and that the view
   sits BELOW it. Frames print innermost-first, so "detected inside the dialect"
   is exactly "dialect4 appears before graph_view".

   Note what the frames actually say: [Shape4.of_vec6] is inlined into
   [Dialect4.validate_sig], so the detection site is attributed to dialect4.ml,
   not shape4.ml. Which frame survives inlining is not predictable in advance —
   hence asserting the relationship between two frames rather than an index. *)
let%expect_test
    "view4: Invalid_sig keeps the detection site of the dialect check" =
  let id = Tensor_id.of_int 0 in
  let graph =
    {
      Graph_common.Graph.nodes = [];
      root =
        {
          Graph_common.Group.id = Graph_common.Group_id.of_int 0;
          label = None;
          items = [];
        };
      tensors =
        Tensor_id.Map.singleton id
          (Tensor_sig.create ~id ~name:"" ~shape:(s 1 1 5 4 4 3)
             ~fmt:(Payload.Fmt Payload.F32) ());
      inputs = [ id ];
      input_kinds = Tensor_id.Map.empty;
      outputs = [ id ];
    }
  in
  match Framework.View4.of_graph graph with
  | Ok _ -> Format.printf "unexpected Ok@."
  | Error e ->
      let trace =
        Core.Pretty.to_string (Err.Error.pp Framework.View4.pp_error) e
      in
      (* Both claims are about backtrace FRAMES, and js_of_ocaml captures none
         (this suite runs under it too -- see the [modes] field in dune). Where
         there are no frames the ordering is unobservable rather than false, so
         assert instead that the "unavailable" branch rendered. Natively the
         real ordering check is unchanged. Same technique as
         test/native/core_test.ml. *)
      let slots =
        match Err.Error.origin e with
        | None -> None
        | Some origin -> (
            match Err.Origin.stack origin with
            | None -> None
            | Some stack -> (
                match Err.Stack.to_raw_backtrace stack with
                | None -> None
                | Some bt -> Printexc.backtrace_slots bt))
      in
      let detected, ordered =
        match slots with
        | Some _ ->
            ( index_of trace "dialect4.ml" <> None,
              match
                (index_of trace "dialect4.ml", index_of trace "graph_view.ml")
              with
              | Some d, Some v -> d < v
              | _ -> false )
        | None ->
            let notice =
              index_of trace "stack unavailable" <> None
              || index_of trace "not linked with -g" <> None
            in
            (notice, notice)
      in
      Format.printf "detected_in_dialect=%b view_is_only_the_caller=%b@."
        detected ordered;
      [%expect {| detected_in_dialect=true view_is_only_the_caller=true |}]
