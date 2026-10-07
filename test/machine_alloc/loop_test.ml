(* Natural loops: back edges to one header make one loop, and a block the
   entry cannot reach starts no back edge. *)

open Machine_ir

let b = Mir_id.Block.of_int

let show graph =
  Mir_loop.find ~entry:(b 0)
    (List.map (fun (x, ss) -> (b x, List.map b ss)) graph)
  |> List.sort (fun (x : Mir_loop.t) (y : Mir_loop.t) ->
      compare
        (Mir_id.Block.to_int x.Mir_loop.header)
        (Mir_id.Block.to_int y.Mir_loop.header))
  |> List.iter (fun (l : Mir_loop.t) ->
      Fmt.pr "header bb%d: %a@."
        (Mir_id.Block.to_int l.Mir_loop.header)
        Fmt.(list ~sep:(any " ") int)
        (Mir_loop.Ids.elements l.Mir_loop.body))

let%expect_test "nested loops, two back edges, an unreachable block" =
  (* bb1 heads bb1-bb4, bb2 heads bb2-bb3; bb4 and bb3 both return to bb1's
     loop; bb6 is unreachable and jumps to the entry *)
  show
    [
      (0, [ 1 ]);
      (1, [ 2; 5 ]);
      (2, [ 3; 4 ]);
      (3, [ 2; 1 ]);
      (4, [ 1 ]);
      (5, []);
      (6, [ 0 ]);
    ];
  [%expect {|
    header bb1: 1 2 3 4
    header bb2: 2 3 |}]
