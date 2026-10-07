(* M10.2: block-local scheduling of the model kernels' selected programs on
   both targets — every schedule within its source order's dependences and
   reverified, what it moves, what linear scan then costs, and each dropped
   dependence class caught. *)

open Machine_ir

module Run
    (T : Mir_sel.TARGET)
    (R : Machine_alloc.Mir_linear_scan.POOL)
    (X : sig
      val name : string

      val select :
        sites:Mir_failure.Site_entry.t array ->
        Mir_verify.Generic.t ->
        Mir_sel.Make(T).Verified.t
    end) =
struct
  module Sch = Machine_alloc.Mir_schedule.Make (T)
  module Ls = Machine_alloc.Mir_linear_scan.Make (T) (R)
  module Policy = Machine_alloc.Mir_schedule.Policy

  let moved (before : Sch.S.Verified.t) (after : Sch.S.Verified.t) =
    let bodies v =
      List.concat_map
        (fun (f : (_, _) Mir_func.t) ->
          List.map
            (fun (b : (_, _) Mir_block.t) ->
              List.map
                (fun (i : _ Mir_instr.t) -> i.Mir_instr.id)
                b.Mir_block.body)
            f.Mir_func.blocks)
        (Sch.S.Verified.selected v).Sch.S.program.Mir_program.funcs
    in
    List.fold_left2
      (fun n a b ->
        List.fold_left2 (fun n x y -> if x <> y then n + 1 else n) n a b)
      0 (bodies before) (bodies after)

  let compare () =
    List.iter
      (fun (name, gen, sites) ->
        let v = X.select ~sites gen in
        match (Sch.schedule Policy.Sink v, Sch.schedule Policy.Reverse v) with
        | Error r, _ | _, Error r ->
            Fmt.pr "%s %s: %a@." X.name name
              Machine_alloc.Mir_schedule.Refusal.pp r
        | Ok s, Ok r ->
            Fmt.pr
              "%s %s: sink moves %d (%a), reverse %d@.  source: %a@.  sink: \
               %a@."
              X.name name (moved v s) Mir_id.Revision.pp
              (Sch.S.Verified.selected s).Sch.S.program.Mir_program.revision
              (moved v r) Machine_alloc.Mir_alloc_stats.pp
              (Machine_alloc.Mir_alloc_stats.of_program (Ls.allocate v))
              Machine_alloc.Mir_alloc_stats.pp
              (Machine_alloc.Mir_alloc_stats.of_program (Ls.allocate s)))
      (Scan_test.kernels ())

  (* Each class dropped from the graph under the reverse order: the check
     refuses, and how often the selected verifier alone rejects the same order.
     The graph keeps a flags writer on the side of a condition range it started
     on, which is stricter than the verifier's rule: on AArch64 every writer the
     reverse order moves lands outside any range, so only the check sees it. *)
  let mutations () =
    List.iter
      (fun (label, mutation) ->
        let caught =
          List.filter_map
            (fun (name, gen, sites) ->
              let v = X.select ~sites gen in
              match Sch.schedule ~mutation Policy.Reverse v with
              | Ok _ -> None
              | Error r ->
                  let verifier =
                    Result.is_error
                      (Err.payload
                         (Sch.S.Verified.verify
                            (Sch.reorder ~mutation Policy.Reverse v)))
                  in
                  Some (name, r, verifier))
            (Scan_test.kernels ())
        in
        match caught with
        | [] -> Fmt.pr "%s %s: NOT CAUGHT@." X.name label
        | (name, r, _) :: _ ->
            Fmt.pr
              "%s %s: caught on %d kernels (verifier alone: %d); first %s: %a@."
              X.name label (List.length caught)
              (List.length (List.filter (fun (_, _, v) -> v) caught))
              name Machine_alloc.Mir_schedule.Refusal.pp r)
      Machine_alloc.Mir_schedule.Mutation.
        [ ("data", Data); ("flags", Flags); ("order", Order) ]
end

module A64 =
  Run (Machine_target_aarch64.A64) (Machine_target_aarch64.A64_regs)
    (struct
      include Scan_test.A64_select_

      let name = "aarch64"
    end)

module X64 =
  Run (Machine_target_x86_64.X64) (Machine_target_x86_64.X64_regs)
    (struct
      include Scan_test.X64_select_

      let name = "x86_64"
    end)

module A64_small =
  Run
    (Machine_target_aarch64.A64)
    (struct
      include Machine_target_aarch64.A64_regs

      let allocatable bank = Scan_test.take 3 (allocatable bank)
    end)
    (struct
      include Scan_test.A64_select_

      let name = "aarch64, 3 registers"
    end)

module X64_small =
  Run
    (Machine_target_x86_64.X64)
    (struct
      include Machine_target_x86_64.X64_regs

      let allocatable bank = Scan_test.take 3 (allocatable bank)
    end)
    (struct
      include Scan_test.X64_select_

      let name = "x86_64, 3 registers"
    end)

let%expect_test "dropped dependences are caught" =
  A64.mutations ();
  X64.mutations ();
  [%expect
    {|
    aarch64 data: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb1: i209 scheduled after i210, which depends on it
    aarch64 flags: caught on 3 kernels (verifier alone: 0); first chain/0: dependence: bb4: i219 scheduled after i231, which depends on it
    aarch64 order: caught on 5 kernels (verifier alone: 5); first chain/0: dependence: bb19: i360 scheduled after i391, which depends on it
    x86_64 data: caught on 6 kernels (verifier alone: 6); first chain/0: dependence: bb1: i209 scheduled after i210, which depends on it
    x86_64 flags: caught on 4 kernels (verifier alone: 3); first chain/0: dependence: bb4: i220 scheduled after i232, which depends on it
    x86_64 order: caught on 5 kernels (verifier alone: 5); first chain/0: dependence: bb19: i364 scheduled after i394, which depends on it |}]

let%expect_test "sink scheduling, then linear scan" =
  A64.compare ();
  X64.compare ();
  A64_small.compare ();
  X64_small.compare ();
  [%expect
    {|
    aarch64 chain/0: sink moves 73 (rev1), reverse 194
      source: 225 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 225 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/1: sink moves 129 (rev1), reverse 203
      source: 214 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 214 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 chain/2: sink moves 34 (rev1), reverse 70
      source: 88 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 88 instructions; 7 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 bmm/0: sink moves 57 (rev1), reverse 113
      source: 128 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      sink: 128 instructions; 9 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 softmax/0: sink moves 58 (rev1), reverse 305
      source: 336 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 3 rematerialized; 0 slots (0 bytes)
      sink: 336 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 3 rematerialized; 0 slots (0 bytes)
    aarch64 sdpa/0: sink moves 89 (rev1), reverse 692
      source: 755 instructions; 56 register moves, 2 stores, 2 loads, 0 slot moves, 16 rematerialized; 1 slots (8 bytes)
      sink: 755 instructions; 56 register moves, 2 stores, 2 loads, 0 slot moves, 16 rematerialized; 1 slots (8 bytes)
    x86_64 chain/0: sink moves 89 (rev1), reverse 184
      source: 227 instructions; 89 register moves, 21 stores, 20 loads, 0 slot moves, 15 rematerialized; 11 slots (44 bytes)
      sink: 227 instructions; 89 register moves, 23 stores, 20 loads, 0 slot moves, 15 rematerialized; 12 slots (52 bytes)
    x86_64 chain/1: sink moves 156 (rev1), reverse 191
      source: 208 instructions; 88 register moves, 5 stores, 5 loads, 0 slot moves, 3 rematerialized; 2 slots (8 bytes)
      sink: 208 instructions; 88 register moves, 8 stores, 3 loads, 0 slot moves, 3 rematerialized; 2 slots (12 bytes)
    x86_64 chain/2: sink moves 41 (rev1), reverse 73
      source: 90 instructions; 35 register moves, 5 stores, 5 loads, 0 slot moves, 3 rematerialized; 2 slots (8 bytes)
      sink: 90 instructions; 35 register moves, 4 stores, 3 loads, 0 slot moves, 3 rematerialized; 2 slots (12 bytes)
    x86_64 bmm/0: sink moves 69 (rev1), reverse 111
      source: 129 instructions; 53 register moves, 10 stores, 10 loads, 0 slot moves, 8 rematerialized; 3 slots (12 bytes)
      sink: 129 instructions; 53 register moves, 8 stores, 7 loads, 0 slot moves, 8 rematerialized; 4 slots (20 bytes)
    x86_64 softmax/0: sink moves 77 (rev1), reverse 304
      source: 346 instructions; 103 register moves, 30 stores, 27 loads, 0 slot moves, 26 rematerialized; 7 slots (40 bytes)
      sink: 346 instructions; 107 register moves, 31 stores, 24 loads, 0 slot moves, 22 rematerialized; 6 slots (36 bytes)
    x86_64 sdpa/0: sink moves 115 (rev1), reverse 679
      source: 763 instructions; 172 register moves, 54 stores, 46 loads, 0 slot moves, 59 rematerialized; 8 slots (48 bytes)
      sink: 763 instructions; 172 register moves, 59 stores, 47 loads, 0 slot moves, 53 rematerialized; 8 slots (48 bytes)
    aarch64, 3 registers chain/0: sink moves 73 (rev1), reverse 194
      source: 225 instructions; 11 register moves, 33 stores, 33 loads, 0 slot moves, 25 rematerialized; 15 slots (72 bytes)
      sink: 225 instructions; 11 register moves, 24 stores, 33 loads, 0 slot moves, 28 rematerialized; 14 slots (68 bytes)
    aarch64, 3 registers chain/1: sink moves 129 (rev1), reverse 203
      source: 214 instructions; 8 register moves, 16 stores, 16 loads, 0 slot moves, 26 rematerialized; 4 slots (20 bytes)
      sink: 214 instructions; 7 register moves, 8 stores, 18 loads, 0 slot moves, 32 rematerialized; 5 slots (28 bytes)
    aarch64, 3 registers chain/2: sink moves 34 (rev1), reverse 70
      source: 88 instructions; 6 register moves, 11 stores, 11 loads, 0 slot moves, 9 rematerialized; 3 slots (12 bytes)
      sink: 88 instructions; 6 register moves, 5 stores, 11 loads, 0 slot moves, 11 rematerialized; 3 slots (12 bytes)
    aarch64, 3 registers bmm/0: sink moves 57 (rev1), reverse 113
      source: 128 instructions; 7 register moves, 15 stores, 17 loads, 0 slot moves, 17 rematerialized; 5 slots (24 bytes)
      sink: 128 instructions; 8 register moves, 7 stores, 18 loads, 0 slot moves, 19 rematerialized; 5 slots (24 bytes)
    aarch64, 3 registers softmax/0: sink moves 58 (rev1), reverse 305
      source: 336 instructions; 11 register moves, 39 stores, 39 loads, 0 slot moves, 28 rematerialized; 9 slots (56 bytes)
      sink: 336 instructions; 11 register moves, 23 stores, 39 loads, 0 slot moves, 31 rematerialized; 9 slots (56 bytes)
    aarch64, 3 registers sdpa/0: sink moves 89 (rev1), reverse 692
      source: 755 instructions; 22 register moves, 80 stores, 79 loads, 0 slot moves, 53 rematerialized; 13 slots (84 bytes)
      sink: 755 instructions; 22 register moves, 62 stores, 81 loads, 0 slot moves, 56 rematerialized; 13 slots (84 bytes)
    x86_64, 3 registers chain/0: sink moves 89 (rev1), reverse 184
      source: 227 instructions; 85 register moves, 59 stores, 56 loads, 0 slot moves, 28 rematerialized; 16 slots (80 bytes)
      sink: 227 instructions; 85 register moves, 43 stores, 40 loads, 0 slot moves, 28 rematerialized; 15 slots (76 bytes)
    x86_64, 3 registers chain/1: sink moves 156 (rev1), reverse 191
      source: 208 instructions; 85 register moves, 54 stores, 48 loads, 0 slot moves, 27 rematerialized; 6 slots (36 bytes)
      sink: 208 instructions; 86 register moves, 34 stores, 28 loads, 0 slot moves, 27 rematerialized; 7 slots (44 bytes)
    x86_64, 3 registers chain/2: sink moves 41 (rev1), reverse 73
      source: 90 instructions; 32 register moves, 22 stores, 20 loads, 0 slot moves, 11 rematerialized; 4 slots (20 bytes)
      sink: 90 instructions; 32 register moves, 14 stores, 12 loads, 0 slot moves, 11 rematerialized; 4 slots (20 bytes)
    x86_64, 3 registers bmm/0: sink moves 69 (rev1), reverse 111
      source: 129 instructions; 48 register moves, 33 stores, 31 loads, 0 slot moves, 18 rematerialized; 6 slots (32 bytes)
      sink: 129 instructions; 48 register moves, 21 stores, 19 loads, 0 slot moves, 18 rematerialized; 6 slots (32 bytes)
    x86_64, 3 registers softmax/0: sink moves 77 (rev1), reverse 304
      source: 346 instructions; 99 register moves, 72 stores, 63 loads, 0 slot moves, 43 rematerialized; 9 slots (56 bytes)
      sink: 346 instructions; 99 register moves, 56 stores, 47 loads, 0 slot moves, 42 rematerialized; 9 slots (56 bytes)
    x86_64, 3 registers sdpa/0: sink moves 115 (rev1), reverse 679
      source: 763 instructions; 158 register moves, 120 stores, 102 loads, 0 slot moves, 74 rematerialized; 11 slots (68 bytes)
      sink: 763 instructions; 158 register moves, 101 stores, 83 loads, 0 slot moves, 73 rematerialized; 11 slots (68 bytes) |}]
