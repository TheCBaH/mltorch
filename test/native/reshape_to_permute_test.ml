(* Recognising a reshape that is only relabelling axes, versus a genuine
   flatten. Split from permute_passes_test.ml. *)

open Permute_pass_fixtures

(* ---- reshape as relabelling ---------------------------------------------- *)

let%expect_test "reshape_to_permute: a pure relabelling" =
  (* C=6 becomes H=6 with nothing else non-unit, so the row-major order is
     untouched and the reshape is the permutation H<-C. *)
  run (Graph_fixtures.reshape_relabel ()) [ Reshape_to_permute.pass ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [C=6] ->[n0]]
      nodes:
        n0: [t1 f32 [H=6 W=1 C=1] ->[n1]] =
          reshape x=t0 params={shape=[H=6 W=1 C=1]}
        n1: [t2 f32 [H=6 W=1 C=1]] = relu x=t1 <-n0
      outputs: [t2 f32 [H=6 W=1 C=1] <-n1]
    after:
      graph
      inputs: [t0 f32 [C=6] ->[n2]]
      nodes:
        n2: [t1 f32 [H=6 W=1 C=1] ->[n1]] = permute x=t0 perm=[H<-C, W<-H, C<-W]
        n1: [t2 f32 [H=6 W=1 C=1]] = relu x=t1 <-n2
      outputs: [t2 f32 [H=6 W=1 C=1] <-n1]
    map:
      values:
        identity
      nodes:
        {n0} -> {n2}
      provenance:
        none |}]

let%expect_test "reshape_to_permute: a genuine flatten is left alone" =
  (* H=2,W=3 collapsing onto C=6 mixes extents; no permutation of the six axes
     computes it. *)
  matches Reshape_to_permute.pattern (Graph_fixtures.reshape_flatten ());
  [%expect {| no match |}]

let%expect_test "reshape_to_permute: the permute computes the same tensor" =
  (* The load-bearing check for this pass. Its legality argument is about
     row-major offsets, not a syntactic identity, so only the numbers confirm
     it — and this case moves TWO non-unit axes ([W=2 C=3] becoming [H=2 W=3]),
     where a plausible-but-wrong perm would still produce the right shape. *)
  let g =
    Graph_builder.build ~name:"relabel2"
      ~outputs:(fun o -> [ o ])
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.s 1 1 1 1 2 3) () in
        reshape { Reshape.Reshape.shape = Graph_fixtures.s 1 1 1 2 3 1 } x)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  let input =
    Tensor.materialize (Graph_fixtures.s 1 1 1 1 2 3) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.W) * 3) + Dim.to_int (Vec6.get c Axis.C)))
  in
  let before = evaluated g [ input ] in
  Format.printf "reshape:  %s@." before;
  (match rewritten g [ Reshape_to_permute.pass ] with
  | None -> Format.printf "rewrite failed@."
  | Some g' ->
      Format.printf "@[<v 2>as permute:@,%a@]@." Graph_ir.pp g';
      let after = evaluated g' [ input ] in
      Format.printf "permute:  %s@." after;
      Format.printf "same:     %b@." (String.equal before after));
  [%expect
    {|
    reshape:  tensor f32 [H=2 W=3 C=1] {0, 1, 2, 3, 4, 5}
    as permute:
      graph
      inputs: [t0 f32 [W=2 C=3] ->[n1]]
      nodes:
        n1: [t1 f32 [H=2 W=3 C=1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      outputs: [t1 f32 [H=2 W=3 C=1] <-n1]
    permute:  tensor f32 [H=2 W=3 C=1] {0, 1, 2, 3, 4, 5}
    same:     true |}]
