(* [Arena_problem.of_script]: the per-kind interval scripts an arena and an
   allocator evaluation both solve. *)

open Graph_ir

let t = Tensor_id.of_int

let sg i fmt =
  Tensor_sig.create ~id:(t i) ~name:"" ~shape:(Graph_fixtures.s1c 4) ~fmt ()

let alloc ~released s =
  match
    Err.payload
      (Alloc_script.alloc
         ~released:(Tensor_id.Set.of_list (List.map t released))
         s)
  with
  | Ok a -> Alloc_script.Event.Alloc a
  | Error _ -> assert false

let node i = Alloc_script.Event.Node (Node_id.of_int i)
let free i = Alloc_script.Event.Free (t i)
let f32 = Payload.Fmt Payload.F32
let i64 = Payload.Fmt Payload.I64

let show script =
  match Err.payload (Arena_problem.of_script script) with
  | Error (`Arena_script id) -> Fmt.pr "inconsistent at %a@." Tensor_id.pp id
  | Ok p ->
      List.iter
        (fun { Arena_problem.Kind_problem.kind; script } ->
          Fmt.pr "%a:" Alloc_script.Kind.pp kind;
          List.iter
            (function
              | Interval_alloc.Event.Alloc { key; size } ->
                  Fmt.pr " +%a(%Ld)" Tensor_id.pp key size
              | Interval_alloc.Event.Free key -> Fmt.pr " -%a" Tensor_id.pp key)
            (Interval_alloc.Script.events script);
          Fmt.pr "@.")
        (Arena_problem.kinds p);
      Fmt.pr "first %a@." Tensor_id.pp (Arena_problem.first_id p)

(* Node markers go, ineligible edges go, and each kind keeps its own events
   in script order: [t1] is freed only after [t3] is allocated, so the two
   still conflict. [t4] is never freed and so stays live to the end. *)
let%expect_test "projection keeps order and drops what is not eligible" =
  let released = [ 1; 2; 3 ] in
  show
    [
      node 0;
      alloc ~released (sg 1 f32);
      alloc ~released (sg 2 i64);
      alloc ~released (sg 5 f32);
      node 1;
      alloc ~released (sg 3 f32);
      free 1;
      free 2;
      node 2;
      alloc ~released:[ 4 ] (sg 4 f32);
      free 3;
    ];
  [%expect
    {|
    float32: +t1(4) +t3(4) -t1 +t4(4) -t3
    int64: +t2(4) -t2
    first t1 |}]

let%expect_test "no eligible edge: no problem" =
  show [ node 0; alloc ~released:[] (sg 1 f32) ];
  [%expect {| first t0 |}]

let%expect_test "an eligible edge allocated twice is inconsistent" =
  let released = [ 1 ] in
  show [ alloc ~released (sg 1 f32); alloc ~released (sg 1 f32) ];
  [%expect {| inconsistent at t1 |}]
