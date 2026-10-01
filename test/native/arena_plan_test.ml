(* [Arena_plan.create]: a dry run's script becomes per-kind pools with a
   witnessed slot for every eligible edge, and nothing else. *)

open Graph_ir
open Core.Storage_units

let units r = Err.or_raise ~pp_error r
let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty
let t = Tensor_id.of_int
let ok = function Ok x -> x | Error _ -> Fmt.failwith "unexpected error"

let dry ?(retain = only_empty) g =
  match Err.payload (Eval_direct.dry_run ~retain g) with
  | Ok s -> s
  | Error _ -> Fmt.failwith "dry run failed"

let pp_error ppf = function
  | `Arena_over_limit
      { Arena_plan.Over_limit.kind; numel; bytes; limit = Bytes limit } ->
      Fmt.pf ppf "over limit: %a pool of %a cells (%a bytes) > %a bytes"
        Arena_plan.Kind.pp kind Element_count.pp numel Byte_size.pp bytes
        Byte_size.pp limit
  | `Arena_over_limit
      { Arena_plan.Over_limit.kind; numel; limit = Elements l; _ } ->
      Fmt.pf ppf "over limit: %a pool of %a cells >= %a cells"
        Arena_plan.Kind.pp kind Element_count.pp numel Element_count.pp l
  | `Arena_placement _ -> Fmt.pf ppf "placement failed"
  | `Arena_script id -> Fmt.pf ppf "inconsistent script at %a" Tensor_id.pp id

let show plan =
  List.iter
    (fun (s : Arena_plan.Slot.t) ->
      Fmt.pr "%a: %a @@%a +%a (%a cells)@." Tensor_id.pp s.id Arena_plan.Kind.pp
        s.kind Byte_offset.pp s.offset Byte_size.pp s.bytes Element_count.pp
        s.numel)
    (Arena_plan.slots plan);
  List.iter
    (fun (p : Arena_plan.Pool.t) ->
      Fmt.pr "pool %a: %a cells, %a bytes, align %a@." Arena_plan.Kind.pp p.kind
        Element_count.pp p.numel Byte_size.pp p.bytes Byte_alignment.pp
        p.alignment)
    (Arena_plan.pools plan);
  let s = Arena_plan.stats plan in
  Fmt.pr "bytes: pool %a, per-kind bound %a, combined bound %a, outside %a@."
    Byte_size.pp s.pool_bytes Byte_size.pp s.per_kind_bound_bytes Byte_size.pp
    s.combined_bound_bytes Byte_size.pp s.out_of_arena_bytes

let create ?limits script =
  match Err.payload (Arena_plan.create ?limits script) with
  | Ok p -> p
  | Error e -> Fmt.failwith "plan: %a" pp_error e

let fixture name = (List.assoc name Graph_fixtures.all) ()

let%expect_test "small graphs" =
  show (create (dry (fixture "residual")));
  Fmt.pr "--@.";
  show (create (dry (fixture "chain")));
  [%expect
    {|
    t1: float32 @0 +16 (4 cells)
    t2: float32 @64 +16 (4 cells)
    pool float32: 20 cells, 80 bytes, align 64
    bytes: pool 80, per-kind bound 128, combined bound 128, outside 16
    --
    t7: float32 @0 +108 (27 cells)
    t8: float32 @128 +108 (27 cells)
    pool float32: 59 cells, 236 bytes, align 64
    bytes: pool 236, per-kind bound 256, combined bound 256, outside 108 |}]

(* Only the released edges are placed: a graph output and an explicitly
   retained edge stay outside, as does an index output nothing reads. *)
let%expect_test "outputs, retained edges and dead index outputs are absent" =
  Fmt.pr "retain t1:@.";
  show
    (create
       (dry
          ~retain:(Release_schedule.Retain.Only (Tensor_id.Set.singleton (t 1)))
          (fixture "residual")));
  Fmt.pr "--@.multi_output:@.";
  show (create (dry (fixture "multi_output")));
  Fmt.pr "--@.retain All:@.";
  show (create (dry ~retain:Release_schedule.Retain.All (fixture "residual")));
  [%expect
    {|
    retain t1:
    t2: float32 @0 +16 (4 cells)
    pool float32: 4 cells, 16 bytes, align 64
    bytes: pool 16, per-kind bound 64, combined bound 64, outside 32
    --
    multi_output:
    t1: float32 @0 +32 (8 cells)
    pool float32: 8 cells, 32 bytes, align 64
    bytes: pool 32, per-kind bound 64, combined bound 64, outside 32
    --
    retain All:
    bytes: pool 0, per-kind bound 0, combined bound 0, outside 48 |}]

(* Hand-built scripts, for what no fixture graph has. *)
let sg ?quant i fmt =
  Tensor_sig.create ~id:(t i) ~name:"" ~shape:(Graph_fixtures.s1c 4) ~fmt ?quant
    ()

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

let%expect_test "several kinds get separate pools" =
  let released = [ 1; 2; 3; 4 ] in
  show
    (create
       [
         node 0;
         alloc ~released (sg 1 f32);
         alloc ~released (sg 2 (Payload.Fmt Payload.I64));
         alloc ~released (sg 3 (Payload.Fmt Payload.Bool));
         node 1;
         alloc ~released (sg 4 f32);
         free 1;
         free 2;
         free 3;
         free 4;
       ]);
  [%expect
    {|
    t1: float32 @0 +16 (4 cells)
    t2: int64 @0 +32 (4 cells)
    t3: int8_unsigned @0 +4 (4 cells)
    t4: float32 @64 +16 (4 cells)
    pool float32: 20 cells, 80 bytes, align 64
    pool int64: 4 cells, 32 bytes, align 64
    pool int8_unsigned: 4 cells, 4 bytes, align 64
    bytes: pool 116, per-kind bound 256, combined bound 256, outside 0 |}]

(* A dead non-index output lives from its node to that node's own release: it
   still conflicts with what the same node allocates. *)
let%expect_test "a dead output occupies its node only" =
  let released = [ 1; 2; 3 ] in
  show
    (create
       [
         node 0;
         alloc ~released (sg 1 f32);
         alloc ~released (sg 2 f32);
         free 1;
         node 1;
         alloc ~released (sg 3 f32);
         free 2;
         free 3;
       ]);
  [%expect
    {|
    t1: float32 @64 +16 (4 cells)
    t2: float32 @0 +16 (4 cells)
    t3: float32 @64 +16 (4 cells)
    pool float32: 20 cells, 80 bytes, align 64
    bytes: pool 80, per-kind bound 128, combined bound 128, outside 0 |}]

let%expect_test "a quantized edge stays outside the arena" =
  let released = [ 1; 2 ] in
  let q = Quant.per_tensor ~scale:0.5 ~zero_point:0 in
  show
    (create
       [
         node 0;
         alloc ~released (sg 1 f32);
         alloc ~released (sg ~quant:q 2 (Payload.Fmt Payload.I8));
         free 1;
         free 2;
       ]);
  [%expect
    {|
    t1: float32 @0 +16 (4 cells)
    pool float32: 4 cells, 16 bytes, align 64
    bytes: pool 16, per-kind bound 64, combined bound 64, outside 4 |}]

let limits ~max_bytes =
  let d = Kernel.Limits.default in
  match
    Err.payload
      (Kernel.Limits.create ~max_size:d.max_size ~max_depth:d.max_depth
         ~max_values:d.max_values ~max_dep_depth:d.max_dep_depth
         ~max_inputs:d.max_inputs ~max_outputs:d.max_outputs
         ~max_extent:d.max_extent ~max_numel:d.max_numel ~max_bytes
         ~max_local_slots:d.max_local_slots ~max_scan_state:d.max_scan_state
         ~max_scan_updates_per_key:d.max_scan_updates_per_key
         ~max_scan_updates_total:d.max_scan_updates_total)
  with
  | Ok l -> l
  | Error _ -> assert false

let%expect_test "a pool over a ceiling is refused, never partly built" =
  let released = [ 1; 2 ] in
  let two_f32 =
    [
      node 0;
      alloc ~released (sg 1 f32);
      alloc ~released (sg 2 f32);
      free 1;
      free 2;
    ]
  in
  (match
     Err.payload (Arena_plan.create ~limits:(limits ~max_bytes:16L) two_f32)
   with
  | Ok _ -> Fmt.pr "planned@."
  | Error e -> Fmt.pr "%a@." pp_error e);
  (match
     Err.payload (Arena_plan.create ~limits:(limits ~max_bytes:128L) two_f32)
   with
  | Ok _ -> Fmt.pr "planned@."
  | Error e -> Fmt.pr "%a@." pp_error e);
  (* Two live edges of 2^30 cells reach the element ceiling (2^31). *)
  let big i =
    Tensor_sig.create ~id:(t i) ~name:""
      ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:(1 lsl 30))
      ~fmt:f32 ()
  in
  (match
     Err.payload
       (Arena_plan.create
          [
            node 0;
            alloc ~released (big 1);
            alloc ~released (big 2);
            free 1;
            free 2;
          ])
   with
  | Ok _ -> Fmt.pr "planned@."
  | Error e -> Fmt.pr "%a@." pp_error e);
  [%expect
    {|
    over limit: float32 pool of 20 cells (80 bytes) > 16 bytes
    planned
    over limit: float32 pool of 2147483648 cells >= 2147483648 cells |}]

(* An independent replay: no two edges live at the same time share cells of one
   pool, and every eligible allocation has a slot inside its pool, at its
   script alignment. *)
let verify plan =
  let slot id = Arena_plan.slot plan id in
  let pool kind =
    List.find_opt
      (fun (p : Arena_plan.Pool.t) -> Arena_plan.Kind.equal p.kind kind)
      (Arena_plan.pools plan)
  in
  let stop (s : Arena_plan.Slot.t) =
    units (Byte_offset.advance s.offset s.bytes)
  and lt a b = Byte_offset.compare a b < 0 in
  let live = ref [] and bad = ref 0 in
  List.iter
    (function
      | Alloc_script.Event.Alloc a when a.Alloc_script.Alloc.eligible -> (
          match slot a.Alloc_script.Alloc.id with
          | None -> incr bad
          | Some s ->
              if
                not
                  (Byte_offset.is_aligned s.offset
                     a.Alloc_script.Alloc.alignment)
              then incr bad;
              (match pool s.kind with
              | Some p
                when Byte_size.compare (Byte_offset.to_size (stop s)) p.bytes
                     <= 0 ->
                  ()
              | _ -> incr bad);
              List.iter
                (fun (o : Arena_plan.Slot.t) ->
                  if
                    Arena_plan.Kind.equal o.kind s.kind
                    && lt o.offset (stop s)
                    && lt s.offset (stop o)
                  then incr bad)
                !live;
              live := s :: !live)
      | Alloc_script.Event.Alloc a ->
          if slot a.Alloc_script.Alloc.id <> None then incr bad
      | Alloc_script.Event.Free id ->
          live :=
            List.filter
              (fun (o : Arena_plan.Slot.t) -> not (Tensor_id.equal o.id id))
              !live
      | Alloc_script.Event.Node _ -> ())
    (Arena_plan.script plan);
  !bad

let%expect_test "every fixture plans, and the replay agrees" =
  List.iter
    (fun (name, build) ->
      let g = build () in
      List.iter
        (fun retain ->
          let plan = create (dry ~retain g) in
          let bad = verify plan in
          if bad <> 0 then Fmt.pr "%s: %d violations@." name bad)
        [ only_empty; Release_schedule.Retain.All ])
    Graph_fixtures.all;
  Fmt.pr "%d fixtures planned@." (List.length Graph_fixtures.all);
  ignore ok;
  [%expect {|
    45 fixtures planned |}]

(* A plan keeps the policy it was made under, and is reused under another only
   through [revalidate]: a stronger policy that some slot's offset fails is
   refused, a weaker one keeps the placement. *)
let%expect_test "revalidate under another alignment policy" =
  let host v = Alignment_policy.with_host (units (Byte_alignment.of_int64 v)) in
  let plan = create (dry (fixture "residual")) in
  let try_policy policy =
    match Err.payload (Arena_plan.revalidate plan policy) with
    | Ok p ->
        Fmt.pr "%a: kept, base %a@." Alignment_policy.pp (Arena_plan.policy p)
          (Fmt.option Byte_alignment.pp)
          (Arena_plan.base_alignment p)
    | Error
        (`Arena_placement
           (`Misaligned
              {
                Interval_alloc.Misaligned.block = { key; offset; _ };
                alignment;
              })) ->
        Fmt.pr "%a: %a at %a is not a multiple of %a@." Alignment_policy.pp
          policy Tensor_id.pp key Byte_offset.pp offset Byte_alignment.pp
          alignment
    | Error e -> Fmt.pr "%a@." pp_error e
  in
  Fmt.pr "made under %a@." Alignment_policy.pp (Arena_plan.policy plan);
  try_policy (host 32L);
  try_policy (host 128L);
  let strong =
    match
      Err.payload
        (Arena_plan.create ~alignment:(host 128L)
           (Err.or_raise ~pp_error:Eval_direct.pp_error
              (Eval_direct.dry_run ~alignment:(host 128L) ~retain:only_empty
                 (fixture "residual"))))
    with
    | Ok p -> p
    | Error e -> Fmt.failwith "plan: %a" pp_error e
  in
  show strong;
  [%expect
    {|
    made under standard
    standard, host 32: kept, base 64
    standard, host 128: t2 at 64 is not a multiple of 128
    t1: float32 @0 +16 (4 cells)
    t2: float32 @128 +16 (4 cells)
    pool float32: 36 cells, 144 bytes, align 128
    bytes: pool 144, per-kind bound 256, combined bound 256, outside 16 |}]
