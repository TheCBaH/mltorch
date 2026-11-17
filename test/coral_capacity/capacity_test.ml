open Coral_capacity

let profile name =
  let text =
    In_channel.with_open_bin "../coralnpu/fragment.json" In_channel.input_all
  in
  match
    Err.payload
      (Err.bind (Coral_manifest.of_string text) (fun m ->
           Coral_profile.of_manifest m name))
  with
  | Ok p -> p
  | Error e -> Fmt.failwith "%a" Coral_profile.pp_error e

let rtl = profile "tiny-rtl"
let mpact = profile "tiny-mpact"

let highmem =
  {
    rtl with
    configuration_id = "coralnpu:tiny-highmem:x";
    regions =
      List.map
        (fun (r : Coral_profile.Region.t) ->
          match r.role with
          | Itcm -> { r with size = 0x100000L }
          | Dtcm -> { r with base = 0x100000L; size = 0x100000L }
          | Csr | Ddr | Mmio | Ram | Rom -> r)
        rtl.regions;
  }

(* The two code sizes the design records: three kernels and the runtime of one
   mobilenetv2_050 block, and all six nodes unfused (the driver is in neither). *)
let fused = Code_bytes.of_int64 4_960L
let unfused = Code_bytes.of_int64 7_008L

let show p image =
  match Err.payload (Capacity.place p image) with
  | Ok (placed : Capacity.Placed.t) ->
      Fmt.pr "%s: code %a@." image.Image.name Code_bytes.pp placed.code;
      List.iter
        (fun (name, (s : Coral_layout.Placed_slot.t)) ->
          Fmt.pr "  %-8s 0x%08Lx %9Ld %s@." name s.address s.slot.size s.region)
        placed.slots
  | Error e -> Fmt.pr "%s: refused: %a@." image.Image.name Capacity.pp_error e

let%expect_test "the witnesses' data sizes" =
  List.iter
    (fun (i : Image.t) ->
      let total =
        List.fold_left
          (fun acc (a : Image.Allocation.t) -> Data_bytes.add acc a.bytes)
          Data_bytes.zero i.allocations
      in
      Fmt.pr "%s: %a@." i.name Data_bytes.pp total;
      List.iter
        (fun (a : Image.Allocation.t) ->
          Fmt.pr "  %-8s %a@." a.name Data_bytes.pp a.bytes)
        i.allocations)
    (Witness.all ~code:fused);
  [%expect
    {|
    mobilenetv2_050 pointwise 8 -> 48 at 112x112: 2852864
      input    401408
      output   2408448
      scratch  43008
    efficientnet_b0 expand 16 -> 96, depthwise 3x3 stride 2: 2265088
      input    802816
      output   1204224
      band     172032
      scratch  86016 |}]

let%expect_test "both witnesses placed on both 8 KiB configurations" =
  List.iter
    (fun p -> List.iter (show p) (Witness.all ~code:fused))
    [ rtl; mpact ];
  [%expect
    {|
    mobilenetv2_050 pointwise 8 -> 48 at 112x112: code 4960
      input    0x20000000    401408 testbench-ram
      output   0x20062000   2408448 testbench-ram
      scratch  0x202ae000     43008 testbench-ram
    efficientnet_b0 expand 16 -> 96, depthwise 3x3 stride 2: code 4960
      input    0x20000000    802816 testbench-ram
      output   0x200c4000   1204224 testbench-ram
      band     0x201ea000    172032 testbench-ram
      scratch  0x20214000     86016 testbench-ram
    mobilenetv2_050 pointwise 8 -> 48 at 112x112: code 4960
      input    0x20000000    401408 extmem
      output   0x20062000   2408448 extmem
      scratch  0x202ae000     43008 extmem
    efficientnet_b0 expand 16 -> 96, depthwise 3x3 stride 2: code 4960
      input    0x20000000    802816 extmem
      output   0x200c4000   1204224 extmem
      band     0x201ea000    172032 extmem
      scratch  0x20214000     86016 extmem |}]

let%expect_test "the budget: equal fits, one over is a typed refusal" =
  let image code = Witness.pointwise_mobilenetv2_050 ~code in
  let budget = Capacity.budget rtl |> Err.payload |> Result.get_ok in
  Fmt.pr "budget %a@." Code_bytes.pp budget;
  let at n = Code_bytes.of_int64 n in
  show rtl (image budget);
  show rtl (image (at 8_193L));
  show highmem (image (at 8_193L));
  (match Err.payload (Capacity.check_code rtl (at 8_193L)) with
  | Error (`Code_over_budget { Capacity.Code_over_budget.budget; needed }) ->
      Fmt.pr "typed: %a over %a@." Code_bytes.pp needed Code_bytes.pp budget
  | Ok () | Error _ -> Fmt.pr "not the typed refusal@.");
  [%expect
    {|
    budget 8192
    mobilenetv2_050 pointwise 8 -> 48 at 112x112: code 8192
      input    0x20000000    401408 testbench-ram
      output   0x20062000   2408448 testbench-ram
      scratch  0x202ae000     43008 testbench-ram
    mobilenetv2_050 pointwise 8 -> 48 at 112x112: refused: code: 8193 bytes needed, 8192 budget
    mobilenetv2_050 pointwise 8 -> 48 at 112x112: code 8193
      input    0x001011c0    401408 dtcm
      output   0x20000000   2408448 testbench-ram
      scratch  0x001631c0     43008 dtcm
    typed: 8193 over 8192 |}]

let%expect_test "data capacity stays the layout's: a smaller RAM refuses" =
  let small =
    {
      rtl with
      regions =
        List.map
          (fun (r : Coral_profile.Region.t) ->
            match r.role with
            | Ram -> { r with size = 0x200000L }
            | Csr | Ddr | Dtcm | Itcm | Mmio | Rom -> r)
          rtl.regions;
    }
  in
  List.iter (show small) (Witness.all ~code:fused);
  [%expect
    {|
    mobilenetv2_050 pointwise 8 -> 48 at 112x112: refused: testbench-ram: 2408448 bytes needed, 1695744 available
    efficientnet_b0 expand 16 -> 96, depthwise 3x3 stride 2: refused: testbench-ram: 172032 bytes needed, 90112 available |}]

let%expect_test "the unfused code size of the probe block fits 8 KiB" =
  show rtl (Witness.pointwise_mobilenetv2_050 ~code:unfused);
  [%expect
    {|
    mobilenetv2_050 pointwise 8 -> 48 at 112x112: code 7008
      input    0x20000000    401408 testbench-ram
      output   0x20062000   2408448 testbench-ram
      scratch  0x202ae000     43008 testbench-ram |}]

let%expect_test "candidates: the over-budget one is rejected, the rest placed" =
  let small = Witness.pointwise_mobilenetv2_050 ~code:fused in
  let big =
    {
      (Witness.expand_depthwise_efficientnet_b0
         ~code:(Code_bytes.of_int64 8_193L))
      with
      name = "big";
    }
  in
  (match Err.payload (Capacity.admit rtl [ big; small ]) with
  | Ok a ->
      List.iter (fun (n, _) -> Fmt.pr "placed %s@." n) a.placed;
      List.iter
        (fun (r : Capacity.Admission.rejection) ->
          Fmt.pr "rejected %s: %a@." r.name Capacity.pp_error r.reason)
        a.rejected
  | Error _ -> Fmt.pr "none fit@.");
  (match Err.payload (Capacity.admit rtl [ big ]) with
  | Error
      (`None_fit [ { Capacity.Admission.name; reason = `Code_over_budget o } ])
    ->
      Fmt.pr "none fit: %s %a over %a@." name Code_bytes.pp o.needed
        Code_bytes.pp o.budget
  | Ok _ | Error _ -> Fmt.pr "not the typed refusal@.");
  [%expect
    {|
    placed mobilenetv2_050 pointwise 8 -> 48 at 112x112
    rejected big: code: 8193 bytes needed, 8192 budget
    none fit: big 8193 over 8192 |}]

(* Three candidates for one segment: [dear] and [cheap] fit, [big] is over the
   ITCM budget; [twin] costs what [cheap] does. Cost is the fitted VME rates
   over a census of f32 operations alone. *)
let census ops =
  {
    Coral_estimate.Coral_census.counts =
      [ (Coral_estimate.Coral_census.Class.F32_op, ops) ];
    contracts = [];
    attends = [];
    bodies = [];
    unknown = 0;
  }

let candidate ?(code = fused) name ops =
  {
    Capacity.Candidate.census = census ops;
    image = { (Witness.pointwise_mobilenetv2_050 ~code) with Image.name };
  }

let core = Coral_estimate.Coral_model.Core.vme ~packed:(fun _ -> false)

let select cands =
  Capacity.select rtl ~core ~rates:Coral_estimate.Coral_rates.vme cands

let%expect_test "select: the cheapest candidate that fits" =
  let cands =
    [
      candidate ~code:(Code_bytes.of_int64 8_193L) "big" 1.;
      candidate "dear" 1000.;
      candidate "twin" 10.;
      candidate "cheap" 10.;
    ]
  in
  (match Err.payload (select cands) with
  | Ok s ->
      Fmt.pr "chose %s (%.2f)@." s.name s.cycles;
      List.iter (fun (n, c) -> Fmt.pr "  %s %.2f@." n c) s.ranked;
      List.iter
        (fun (r : Capacity.Admission.rejection) ->
          Fmt.pr "rejected %s: %a@." r.name Capacity.pp_error r.reason)
        s.rejected
  | Error _ -> Fmt.pr "none fit@.");
  [%expect
    {|
    chose cheap (47.11)
      cheap 47.11
      twin 47.11
      dear 4711.00
    rejected big: code: 8193 bytes needed, 8192 budget |}];
  match Err.payload (select [ List.hd cands ]) with
  | Error (`None_fit [ { Capacity.Admission.name; _ } ]) ->
      Fmt.pr "none fit: %s@." name;
      [%expect {| none fit: big |}]
  | Ok _ | Error _ -> Fmt.pr "not the typed refusal@."
