open Ssa_ir
open Ssa_fixtures
module B = Ssa_builder

(* The control-flow graph form: that a structured program means the same in it,
   that the printer shows the loop and branch shapes, and that the verifier
   rejects each way a graph can be wrong. *)

let row_buffer id n format role = buffer id ~h:1L ~w:n format role
let idx bld n = B.index bld (Int64.of_int n)

let load_at bld id i =
  B.load_f64 bld (buf id) ~decode:Ssa_op.Decode.F32_to_f64
    (at bld ~h:(idx bld 0) ~w:i)

let store_at bld id i x =
  B.store_f64 bld (buf id) ~encode:Ssa_op.Encode.F32_round
    (at bld ~h:(idx bld 0) ~w:i)
    x

let bufs =
  [
    row_buffer 0 4L Ssa_format.F32 Ssa_buffer.Input;
    row_buffer 1 8L Ssa_format.F32 Ssa_buffer.Output;
  ]

let cfg_of p =
  match Ssa_cfg_lower.program p with
  | Ok c -> c
  | Error e -> Fmt.failwith "%a" Ssa_cfg_lower.pp_error e

let result_text = function
  | Ok () -> "ok"
  | Error e -> Fmt.str "%a" Ssa_interp.pp_error e

(* One program under both executors: the outcome, the output cells and the
   logical work must be the same. *)
let both ?(input = Array.make 4 1.) p =
  let outcome engine =
    let out = Array.make 8 0. in
    let memory = memory [ (0, floats input); (1, floats out) ] in
    let counters = Ssa_interp.Counters.create () in
    let text =
      match engine with
      | `Structured ->
          result_text (Err.payload (Ssa_interp.run ~counters p ~memory))
      | `Cfg -> (
          match
            Err.payload (Ssa_cfg_interp.run ~counters (cfg_of p) ~memory)
          with
          | Ok () -> "ok"
          | Error (#Ssa_interp.failure as f) ->
              Fmt.str "%a" Ssa_interp.pp_error (f :> Ssa_interp.error)
          | Error (`Invalid_cfg _ as e) ->
              Fmt.str "INVALID %a" Ssa_cfg_verify.pp_error e)
    in
    ( text,
      out,
      List.map (Ssa_interp.Counters.mark counters) Ssa_mark.all,
      Ssa_interp.Counters.loads counters )
  in
  let s = outcome `Structured and c = outcome `Cfg in
  let text, out, marks, loads = c in
  Fmt.pr "%s | out %a | marks %a | loads %d | same as structured: %b@." text
    Fmt.(array ~sep:(any " ") float)
    out
    Fmt.(list ~sep:(any ",") int)
    marks loads (s = c)

let%expect_test "recurrences, sums and branches mean the same in a graph" =
  (* (a, b) <- (b, a + b), ten times: an edge that rebinds both at once *)
  both
    (build ~buffers:bufs (fun bld ->
         let (B.Cons (a, B.Cons (b, B.Nil))) =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 10)
             ~init:(B.Cons (B.f64 bld 0., B.Cons (B.f64 bld 1., B.Nil)))
             (fun bld _ (B.Cons (a, B.Cons (b, B.Nil))) ->
               B.Cons (b, B.Cons (B.f64_binary bld Expr.Value.Add a b, B.Nil)))
         in
         store_at bld 1 (idx bld 0) a;
         store_at bld 1 (idx bld 1) b));
  (* a swap, an odd number of times *)
  both
    (build ~buffers:bufs (fun bld ->
         let (B.Cons (a, B.Cons (b, B.Nil))) =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 5)
             ~init:(B.Cons (B.f64 bld 1., B.Cons (B.f64 bld 2., B.Nil)))
             (fun _ _ (B.Cons (a, B.Cons (b, B.Nil))) ->
               B.Cons (b, B.Cons (a, B.Nil)))
         in
         store_at bld 1 (idx bld 0) a;
         store_at bld 1 (idx bld 1) b));
  (* an ordered sum, in order, with one mark per term *)
  both
    (build ~buffers:bufs (fun bld ->
         let sum =
           B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 4)
             ~seed:(B.f64 bld 0.5) (fun bld k ->
               B.mark bld Ssa_mark.Reduction;
               load_at bld 0 k)
         in
         store_at bld 1 (idx bld 0) sum));
  (* a nest: the inner loop's carried value is the outer's *)
  both
    (build ~buffers:bufs (fun bld ->
         let (B.Cons (x, B.Nil)) =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 3)
             ~init:(B.Cons (B.f64 bld 0., B.Nil))
             (fun bld _ (B.Cons (x, B.Nil)) ->
               B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 4)
                 ~init:(B.Cons (x, B.Nil))
                 (fun bld k (B.Cons (y, B.Nil)) ->
                   B.Cons
                     (B.f64_binary bld Expr.Value.Add y (load_at bld 0 k), B.Nil)))
         in
         store_at bld 1 (idx bld 0) x));
  (* a branch inside a loop *)
  both
    (build ~buffers:bufs (fun bld ->
         let (B.Cons (x, B.Nil)) =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 4)
             ~init:(B.Cons (B.f64 bld 0., B.Nil))
             (fun bld k (B.Cons (acc, B.Nil)) ->
               let odd =
                 B.index_compare bld Ssa_op.Compare.Lt (idx bld 1)
                   (B.index_add bld k (idx bld 0))
               in
               B.if_ bld odd
                 ~then_:(fun bld ->
                   B.Cons
                     (B.f64_binary bld Expr.Value.Add acc (B.f64 bld 10.), B.Nil))
                 ~else_:(fun _ -> B.Cons (acc, B.Nil)))
         in
         store_at bld 1 (idx bld 0) x));
  [%expect
    {|
    ok | out 55 89 0 0 0 0 0 0 | marks 0,0,0,0,0,0 | loads 0 | same as structured: true
    ok | out 2 1 0 0 0 0 0 0 | marks 0,0,0,0,0,0 | loads 0 | same as structured: true
    ok | out 4.5 0 0 0 0 0 0 0 | marks 0,0,0,4,0,0 | loads 4 | same as structured: true
    ok | out 12 0 0 0 0 0 0 0 | marks 0,0,0,0,0,0 | loads 12 | same as structured: true
    ok | out 20 0 0 0 0 0 0 0 | marks 0,0,0,0,0,0 | loads 0 | same as structured: true |}]

let%expect_test "zero-trip loops go from entry to exit, and failures leave" =
  let program ~lo ~hi =
    build ~buffers:bufs (fun bld ->
        let seed = B.f64 bld 7. in
        let sum =
          B.ordered_sum bld ~lo:(idx bld lo) ~hi:(idx bld hi) ~seed
            (fun bld _ ->
              B.mark bld Ssa_mark.Reduction;
              load_at bld 0 (idx bld 1000))
        in
        let (B.Cons (carried, B.Nil)) =
          B.for_ bld ~lo:(idx bld lo) ~hi:(idx bld hi)
            ~init:(B.Cons (B.f64 bld 3., B.Nil))
            (fun bld _ (B.Cons (x, B.Nil)) ->
              B.Cons
                ( B.f64_binary bld Expr.Value.Add x
                    (load_at bld 0 (idx bld 1000)),
                  B.Nil ))
        in
        store_at bld 1 (idx bld 0) sum;
        store_at bld 1 (idx bld 1) carried)
  in
  (* the body reads far outside the input: only an executed iteration fails *)
  both (program ~lo:0 ~hi:0);
  both (program ~lo:5 ~hi:2);
  both (program ~lo:0 ~hi:1);
  (* only the selected branch runs *)
  let branch cond =
    build ~buffers:bufs (fun bld ->
        let (B.Cons (x, B.Nil)) =
          B.if_ bld (B.pred bld cond)
            ~then_:(fun bld -> B.Cons (load_at bld 0 (idx bld 100), B.Nil))
            ~else_:(fun bld -> B.Cons (B.f64 bld 5., B.Nil))
        in
        store_at bld 1 (idx bld 0) x)
  in
  both (branch false);
  both (branch true);
  [%expect
    {|
    ok | out 7 3 0 0 0 0 0 0 | marks 0,0,0,0,0,0 | loads 0 | same as structured: true
    ok | out 7 3 0 0 0 0 0 0 | marks 0,0,0,0,0,0 | loads 0 | same as structured: true
    t0: coordinate W = 1000 is outside the buffer, at (0,0,0,0,1000,0) | out 0 0 0 0 0 0 0 0 | marks 0,0,0,1,0,0 | loads 0 | same as structured: true
    ok | out 5 0 0 0 0 0 0 0 | marks 0,0,0,0,0,0 | loads 0 | same as structured: true
    t0: coordinate W = 100 is outside the buffer, at (0,0,0,0,100,0) | out 0 0 0 0 0 0 0 0 | marks 0,0,0,0,0,0 | loads 0 | same as structured: true |}]

let%expect_test "the graph of a loop and of a branch" =
  let p =
    build ~buffers:bufs (fun bld ->
        let (B.Cons (x, B.Nil)) =
          B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 3)
            ~init:(B.Cons (B.f64 bld 0., B.Nil))
            (fun bld k (B.Cons (acc, B.Nil)) ->
              let c = B.index_compare bld Ssa_op.Compare.Lt k (idx bld 1) in
              B.if_ bld c
                ~then_:(fun bld ->
                  B.Cons
                    ( B.f64_binary bld Expr.Value.Add acc (load_at bld 0 k),
                      B.Nil ))
                ~else_:(fun _ -> B.Cons (acc, B.Nil)))
        in
        store_at bld 1 (idx bld 0) x)
  in
  Fmt.pr "%a" Ssa_cfg_pp.pp (cfg_of p);
  [%expect
    {|
    buffer b0 in f32 [1, 1, 1, 1, 4, 1]
    buffer b1 out f32 [1, 1, 1, 1, 8, 1]
    bb0(%0:effect):
      (%1:f64) = const 0x0p+0:f64
      (%2:index) = const 3:index
      (%3:index) = const 0:index
      jump bb1(%3, %1, %0)
    bb1(%4:index, %5:f64, %6:effect):
      (%7:pred) = index.compare.lt %4, %2
      branch %7, bb3(), bb2(%5, %6)
    bb2(%8:f64, %9:effect):
      (%10:index) = const 0:index
      (%11:index) = const 0:index
      (%12:index) = const 0:index
      (%13:effect) = store.f32_round b1[%12, %12, %12, %11, %10, %12], %8 effect %9
      return %13
    bb3():
      (%14:index) = const 1:index
      (%15:pred) = index.compare.lt %4, %14
      branch %15, bb5(), bb4()
    bb4():
      jump bb6(%5, %6)
    bb5():
      (%16:index) = const 0:index
      (%17:index) = const 0:index
      (%18:f64, %19:effect) = load.f32_to_f64 b0[%17, %17, %17, %16, %4, %17] effect %6
      (%20:f64) = float.add %5, %18
      jump bb6(%20, %19)
    bb6(%21:f64, %22:effect):
      (%23:index) = const 1:index
      (%24:index) = index.add_in_domain %4, %23
      jump bb1(%24, %21, %22) |}]

(* ---- the verifier rejects each way a graph can be wrong --------------------- *)

let sample =
  cfg_of
    (build ~buffers:bufs (fun bld ->
         let (B.Cons (x, B.Nil)) =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 3)
             ~init:(B.Cons (B.f64 bld 0., B.Nil))
             (fun bld k (B.Cons (acc, B.Nil)) ->
               let c = B.index_compare bld Ssa_op.Compare.Lt k (idx bld 1) in
               B.if_ bld c
                 ~then_:(fun bld ->
                   B.Cons
                     ( B.f64_binary bld Expr.Value.Add acc (load_at bld 0 k),
                       B.Nil ))
                 ~else_:(fun _ -> B.Cons (acc, B.Nil)))
         in
         store_at bld 1 (idx bld 0) x))

let show name (cfg : Ssa_cfg.t) =
  match Err.payload (Ssa_cfg_verify.check cfg) with
  | Ok () -> Fmt.pr "%s: accepted@." name
  | Error e -> Fmt.pr "%s: %a@." name Ssa_cfg_verify.pp_error e

let map_block cfg ~at f =
  {
    cfg with
    Ssa_cfg.blocks =
      List.mapi (fun i b -> if i = at then f b else b) cfg.Ssa_cfg.blocks;
  }

let block cfg i = List.nth cfg.Ssa_cfg.blocks i

(* the blocks as the lowering finishes them: 0 entry, 1 loop header, 2 body,
   3 branch arm, 4 other arm, 5 join, 6 exit *)
let with_terminator cfg ~at terminator =
  map_block cfg ~at (fun b -> { b with Ssa_cfg_block.terminator })

let edge_of = function
  | Ssa_cfg_terminator.Jump e -> e
  | _ -> invalid_arg "a jump"

let%expect_test "a graph that is not well formed is refused" =
  show "the graph as lowered" sample;
  (* the loop header's back edge drops its carried value *)
  let back = block sample 5 in
  let e = edge_of back.Ssa_cfg_block.terminator in
  show "short edge"
    (with_terminator sample ~at:5
       (Ssa_cfg_terminator.Jump { e with args = List.tl e.args }));
  (* the carried float and the induction value trade places *)
  show "retyped edge"
    (with_terminator sample ~at:5
       (Ssa_cfg_terminator.Jump { e with args = List.rev e.args }));
  (* the back edge passes the header's own effect instead of the chain's *)
  let header = block sample 1 in
  let header_effect = List.nth header.Ssa_cfg_block.params 2 in
  show "stale effect"
    (with_terminator sample ~at:5
       (Ssa_cfg_terminator.Jump
          {
            e with
            args = [ List.nth e.args 0; List.nth e.args 1; header_effect ];
          }));
  (* a target that does not exist *)
  show "unknown target"
    (with_terminator sample ~at:5
       (Ssa_cfg_terminator.Jump { e with target = Ssa_id.Block.of_int 999 }));
  (* the entry is jumped back to *)
  show "entry with a predecessor"
    (with_terminator sample ~at:5
       (Ssa_cfg_terminator.Jump
          {
            target = sample.Ssa_cfg.entry;
            args = [ List.hd (block sample 0).Ssa_cfg_block.params ];
          }));
  (* a branch on something that is not a predicate *)
  let loop_branch = block sample 1 in
  (match loop_branch.Ssa_cfg_block.terminator with
  | Ssa_cfg_terminator.Branch b ->
      show "branch on an index"
        (with_terminator sample ~at:1
           (Ssa_cfg_terminator.Branch
              { b with cond = List.hd header.Ssa_cfg_block.params }))
  | _ -> ());
  (* a block nothing reaches *)
  show "unreachable block"
    {
      sample with
      Ssa_cfg.blocks =
        sample.Ssa_cfg.blocks
        @ [
            {
              Ssa_cfg_block.id = Ssa_id.Block.of_int 500;
              params = [];
              body = [];
              terminator =
                Ssa_cfg_terminator.Return
                  (List.hd (block sample 0).Ssa_cfg_block.params);
            };
          ];
    };
  (* the same block twice *)
  show "block twice"
    { sample with Ssa_cfg.blocks = sample.Ssa_cfg.blocks @ [ block sample 3 ] };
  (* a value used where its definition does not dominate: the exit block reads
     what only one arm of the branch defines *)
  let arm = block sample 3 in
  let arm_value =
    List.hd (List.hd (List.rev arm.Ssa_cfg_block.body)).Ssa_instr.results
  in
  let exit_block = block sample 6 in
  (match exit_block.Ssa_cfg_block.terminator with
  | Ssa_cfg_terminator.Return _ -> ()
  | _ -> ());
  show "use not dominated"
    (map_block sample ~at:6 (fun b ->
         {
           b with
           Ssa_cfg_block.body =
             List.map
               (fun (i : Ssa_instr.t) ->
                 match i.Ssa_instr.op with
                 | Ssa_op.Store s ->
                     {
                       i with
                       Ssa_instr.op = Ssa_op.Store { s with value = arm_value };
                     }
                 | _ -> i)
               b.Ssa_cfg_block.body;
         }));
  (* a value defined twice *)
  show "definition twice"
    (map_block sample ~at:6 (fun b ->
         {
           b with
           Ssa_cfg_block.body =
             b.Ssa_cfg_block.body
             @ [ List.hd (block sample 0).Ssa_cfg_block.body ];
         }));
  (* a block with two predecessors and no effect parameter: the other arm
     falls into this one *)
  show "merge without an effect parameter"
    (with_terminator sample ~at:4
       (Ssa_cfg_terminator.Jump
          { target = (block sample 3).Ssa_cfg_block.id; args = [] }));
  [%expect
    {|
    the graph as lowered: accepted
    short edge: bb6: an edge passes (f64, effect) to parameters (index, f64, effect)
    retyped edge: bb6: an edge passes (effect, f64, index) to parameters (index, f64, effect)
    stale effect: bb6: effect v6 is not the current effect v15
    unknown target: bb6: block bb999 does not exist
    entry with a predecessor: bb0: the entry block has a predecessor
    branch on an index: bb1: a branch condition is a predicate, not index
    unreachable block: bb500: the block cannot be reached from the entry
    block twice: bb4: the block appears twice
    use not dominated: bb3: v13 is not defined on every path to its use
    definition twice: bb3: v1 is defined twice
    merge without an effect parameter: bb4: a block has one effect parameter, or none and one predecessor; parameters () |}]

(* ---- the handoff: block arguments as parallel copies ------------------------ *)

let%expect_test "the copies of an edge leave out effects and self moves" =
  let show_edge name (e : Ssa_cfg_edge.t) =
    Fmt.pr "%s:" name;
    List.iter
      (fun (m : Ssa_cfg_handoff.Move.t) ->
        Fmt.pr " %a<-%a" Ssa_id.Value.pp m.destination.Ssa_value.id
          Ssa_id.Value.pp m.source.Ssa_value.id)
      (Ssa_cfg_handoff.copies sample e);
    Fmt.pr "@."
  in
  List.iter
    (fun (b : Ssa_cfg_block.t) ->
      List.iteri
        (fun i e -> show_edge (Fmt.str "%a/%d" Ssa_id.Block.pp b.id i) e)
        (Ssa_cfg_terminator.edges b.terminator))
    sample.Ssa_cfg.blocks;
  Fmt.pr "critical edges: %d@." (List.length (Ssa_cfg.critical_edges sample));
  [%expect
    {|
    bb0/0: v4<-v3 v5<-v1
    bb1/0:
    bb1/1: v16<-v5
    bb2/0:
    bb2/1:
    bb4/0: v14<-v13
    bb5/0: v14<-v5
    bb6/0: v4<-v24 v5<-v14
    critical edges: 0 |}]

let%expect_test "a parallel copy is ordered so every source is read first" =
  let value i =
    { Ssa_value.id = Ssa_id.Value.of_int i; ty = Ssa_type.Scalar Ssa_type.F64 }
  in
  (* every assignment of sources to three destinations among four values,
     executed in the order given, against the simultaneous reading *)
  let n = 3 and m = 4 in
  let total = ref 0 and wrong = ref 0 and temps = ref 0 in
  let rec assignments k = function
    | 0 -> [ [] ]
    | d ->
        List.concat_map
          (fun rest -> List.init k (fun s -> s :: rest))
          (assignments k (d - 1))
  in
  List.iter
    (fun sources ->
      incr total;
      let moves =
        List.mapi
          (fun d s ->
            { Ssa_cfg_handoff.Move.destination = value d; source = value s })
          sources
        |> List.filter (fun (m : Ssa_cfg_handoff.Move.t) ->
            not (Ssa_value.equal m.destination m.source))
      in
      let next = ref 100 in
      let fresh _ =
        incr temps;
        incr next;
        value !next
      in
      let ordered = Ssa_cfg_handoff.sequentialize ~fresh moves in
      let env = Hashtbl.create 8 in
      for i = 0 to m - 1 do
        Hashtbl.replace env i (float_of_int (i + 1))
      done;
      let before = Hashtbl.copy env in
      List.iter
        (fun (mv : Ssa_cfg_handoff.Move.t) ->
          let get (v : Ssa_value.t) =
            Hashtbl.find env (v.Ssa_value.id :> int)
          in
          Hashtbl.replace env
            (mv.destination.Ssa_value.id :> int)
            (get mv.source))
        ordered;
      List.iteri
        (fun d s ->
          if Hashtbl.find env d <> Hashtbl.find before s then incr wrong)
        sources)
    (assignments m n);
  Fmt.pr "%d copies, %d wrong, %d temporaries@." !total !wrong !temps;
  [%expect {| 64 copies, 0 wrong, 14 temporaries |}]
