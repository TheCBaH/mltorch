(* [Me_flow]: the rooted-DAG contract (.ai/model_explorer_design.md).

   Every fixture is built by MUTATING a valid spine, because that is the only
   way each rule gets its own witness: a suite of independently hand-written
   invalid spines tends to violate three rules at once and then proves only
   that the first check fires.

   The valid spine below is the real shape — PT2 imported to Native, two Native
   passes and a pack, then the branch at canonical Native to both Native4D and
   the symbolic stages — so "states = transitions + 1" is false of it, which is
   the point. *)

module F = Me_flow
module L = Me_ids.Layer

let limits = Me_limits.Limits.untrusted

let pp_ok ppf r =
  Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:F.pp_error ppf r

let check label flow =
  Format.printf "%-26s %a@." label pp_ok (F.validate ~limits flow)

(* --- the spine --- *)

let state ?produced_by id layer label =
  { F.State.id; graph = "g" ^ id; layer; label; produced_by }

let exec layer leaf index =
  { F.Pass_execution.layer; exec = { Pass.Exec_id.frames = []; leaf; index } }

let transition ?comparison id before after kind =
  { F.Transition.id; before; after; kind; comparison }

(* PT2 ──import──> native/000 ──pass──> 001 ──pass──> 002 ──pack──> 003
                                                              ├─cross_dialect─> native4d/000
                                                              └─adapt────────> symbolic/000 ─adapt─> kernel/000 *)
let valid =
  {
    F.states =
      [
        state "s/pt2/000" L.Pt2 "exported program";
        state ~produced_by:"t/native/000" "s/native/000" L.Native "initial";
        state ~produced_by:"t/native/001" "s/native/001" L.Native "after fold";
        state ~produced_by:"t/native/002" "s/native/002" L.Native
          "after relayout";
        state ~produced_by:"t/native/003" "s/native/003" L.Native "canonical";
        state ~produced_by:"t/native4d/000" "s/native4d/000" L.Native4d "4d";
        state ~produced_by:"t/symbolic/000" "s/symbolic/000" L.Symbolic "stages";
        state ~produced_by:"t/kernel/000" "s/kernel/000" L.Kernel "kernel";
      ];
    transitions =
      [
        transition "t/native/000" "s/pt2/000" "s/native/000" F.Transition.Import;
        transition ~comparison:"c/000" "t/native/001" "s/native/000"
          "s/native/001"
          (F.Transition.Pass (exec L.Native "fold" 0L));
        transition ~comparison:"c/001" "t/native/002" "s/native/001"
          "s/native/002"
          (F.Transition.Pass (exec L.Native "relayout" 1L));
        transition "t/native/003" "s/native/002" "s/native/003"
          F.Transition.Pack;
        transition "t/native4d/000" "s/native/003" "s/native4d/000"
          F.Transition.Cross_dialect;
        transition "t/symbolic/000" "s/native/003" "s/symbolic/000"
          F.Transition.Adapt;
        transition "t/kernel/000" "s/symbolic/000" "s/kernel/000"
          F.Transition.Adapt;
      ];
    graph = "g/flow";
  }

let%expect_test "the real shape validates, and it is not a chain" =
  check "valid" valid;
  Printf.printf "%d states, %d transitions\n"
    (List.length valid.F.states)
    (List.length valid.F.transitions);
  [%expect {|
    valid                      ok
    8 states, 7 transitions |}]

let%expect_test "the zero-changed-pass run still has a Native state" =
  (* The case the spine exists for: [Pass.Exec_id] does not exist for an import
     and there is no execution at all when nothing changed, so a spine derived
     from executions would have no Native state to show. *)
  let flow =
    {
      F.states =
        [
          state "s/pt2/000" L.Pt2 "exported program";
          state ~produced_by:"t/native/000" "s/native/000" L.Native
            "initial = canonical";
        ];
      transitions =
        [
          transition "t/native/000" "s/pt2/000" "s/native/000"
            F.Transition.Import;
        ];
      graph = "g/flow";
    }
  in
  check "import only" flow;
  [%expect {| import only                ok |}]

let%expect_test "a synthetic Native4D pass, which no real model produces" =
  (* v1 adds no Native4D pass — [Native4d.Lower.convert] is the whole path, so
     [s/native4d/000] is a leaf on any real model. The qualified join is only
     meaningful across two specializations, so it is exercised here and nowhere
     that claims to be a model. *)
  let flow =
    {
      valid with
      F.states =
        valid.F.states
        @ [
            state ~produced_by:"t/native4d/001" "s/native4d/001" L.Native4d
              "after a synthetic pass";
          ];
      transitions =
        valid.F.transitions
        @ [
            transition "t/native4d/001" "s/native4d/000" "s/native4d/001"
              (F.Transition.Pass (exec L.Native4d "fold" 0L));
          ];
    }
  in
  check "synthetic Pass4" flow;
  [%expect {| synthetic Pass4            ok |}]

let%expect_test "equal ordinals in two layers are two transitions" =
  (* [Pass.Exec_id]'s ordinal is dense PER specialization, so the Native and
     Native4D executions above are both [fold #0]. Qualifying by layer is what
     keeps them distinct; without it the previous fixture would be a duplicate
     execution. *)
  Format.printf "native   %a@." F.Pass_execution.pp (exec L.Native "fold" 0L);
  Format.printf "native4d %a@." F.Pass_execution.pp (exec L.Native4d "fold" 0L);
  Printf.printf "compare = %d\n"
    (F.Pass_execution.compare (exec L.Native "fold" 0L)
       (exec L.Native4d "fold" 0L));
  [%expect
    {|
    native   native:fold#0
    native4d native4d:fold#0
    compare = -1 |}]

(* --- the mutations that must go red --- *)

let%expect_test "acyclicity is implied, and the walk is guarded anyway" =
  (* [`Cycle] cannot be reached through the rules above it, and this records
     why rather than pretending it has a witness.

     One root, at most one producer per state, and every state reachable from
     that root together make the reachable subgraph a TREE: a state on a cycle
     that is reachable from the root needs an entry edge from outside the
     cycle, which is a second producer. So a cycle can only exist in an
     unreachable component — which the reachability check reports, as both
     fixtures below show.

     The grey/black colouring stays regardless, because it is what makes the
     walk TERMINATE. Relying on an earlier check to guarantee that is fragile
     in the one direction that matters: reorder the producer check after the
     walk and the failure is a hung process, not a wrong answer. *)
  let back_edge =
    {
      valid with
      F.transitions =
        valid.F.transitions
        @ [
            transition "t/native/009" "s/native/003" "s/native/001"
              (F.Transition.Pass (exec L.Native "loop" 9L));
          ];
    }
  in
  check "back edge" back_edge;
  let disconnected_cycle =
    {
      F.states =
        [
          state "s/pt2/000" L.Pt2 "root";
          state ~produced_by:"t/native/000" "s/native/000" L.Native "reachable";
          state ~produced_by:"t/native/002" "s/native/001" L.Native "on a cycle";
        ];
      transitions =
        [
          transition "t/native/000" "s/pt2/000" "s/native/000"
            F.Transition.Import;
          transition "t/native/002" "s/native/001" "s/native/001"
            (F.Transition.Pass (exec L.Native "self" 2L));
        ];
      graph = "g/flow";
    }
  in
  check "disconnected cycle" disconnected_cycle;
  [%expect
    {|
    back edge                  state s/native/001 has more than one producer
    disconnected cycle         state s/native/001 is unreachable from the root |}]

let%expect_test "an orphan state" =
  let flow =
    {
      valid with
      F.states = valid.F.states @ [ state "s/native/009" L.Native "detached" ];
    }
  in
  (* Not produced by anything and not the root, so it is unreachable — and it
     is NOT a second root, because a root must be [Pt2]. *)
  check "orphan" flow;
  [%expect
    {| orphan                     state s/native/009 is unreachable from the root |}]

let%expect_test "a duplicate producer" =
  let flow =
    {
      valid with
      F.transitions =
        valid.F.transitions
        @ [
            transition "t/native/009" "s/native/000" "s/native/002"
              (F.Transition.Pass (exec L.Native "again" 9L));
          ];
    }
  in
  check "two producers" flow;
  [%expect
    {| two producers              state s/native/002 has more than one producer |}]

let%expect_test "a producer the state does not name" =
  let flow =
    {
      valid with
      F.states =
        List.map
          (fun (s : F.State.t) ->
            if String.equal s.F.State.id "s/native/001" then
              { s with F.State.produced_by = Some "t/native/003" }
            else s)
          valid.F.states;
    }
  in
  check "disagreeing producer" flow;
  [%expect
    {| disagreeing producer       state s/native/001 names a producer that does not produce it |}]

let%expect_test "illegal cross-layer transitions" =
  let illegal label before after kind =
    let flow =
      {
        valid with
        F.transitions =
          [
            List.hd valid.F.transitions; transition "t/x/000" before after kind;
          ];
      }
    in
    (* Only the import and the offending transition, so nothing else fires
       first; the unreachable states that leaves are not reached because the
       triple check runs before the walk. *)
    check label flow
  in
  illegal "pt2 -> native4d" "s/pt2/000" "s/native4d/000"
    F.Transition.Cross_dialect;
  illegal "native -> kernel" "s/native/000" "s/kernel/000" F.Transition.Adapt;
  illegal "native4d -> symbolic" "s/native4d/000" "s/symbolic/000"
    F.Transition.Adapt;
  illegal "import inside native" "s/native/000" "s/native/001"
    F.Transition.Import;
  [%expect
    {|
    pt2 -> native4d            transition t/x/000 crosses layers illegally
    native -> kernel           transition t/x/000 crosses layers illegally
    native4d -> symbolic       transition t/x/000 crosses layers illegally
    import inside native       transition t/x/000 crosses layers illegally |}]

let%expect_test "a pass carrying another layer's execution" =
  (* The triple is legal — Native to Native — and the execution is from the
     wrong dialect. Only comparing [e.layer] against the endpoints catches
     it. *)
  let flow =
    {
      valid with
      F.transitions =
        List.map
          (fun (t : F.Transition.t) ->
            if String.equal t.F.Transition.id "t/native/001" then
              {
                t with
                F.Transition.kind =
                  F.Transition.Pass (exec L.Native4d "fold" 0L);
              }
            else t)
          valid.F.transitions;
    }
  in
  check "wrong layer" flow;
  [%expect
    {| wrong layer                transition t/native/001 carries an execution from another layer |}]

let%expect_test "the same execution twice" =
  let flow =
    {
      valid with
      F.transitions =
        List.map
          (fun (t : F.Transition.t) ->
            if String.equal t.F.Transition.id "t/native/002" then
              {
                t with
                F.Transition.kind = F.Transition.Pass (exec L.Native "fold" 0L);
              }
            else t)
          valid.F.transitions;
    }
  in
  check "duplicate execution" flow;
  [%expect
    {| duplicate execution        pass execution native:fold#0 occurs more than once |}]

let%expect_test "roots" =
  (* Dropping the root STATE alone only leaves a dangling endpoint, which is a
     different rule and fires first. To reach [`No_root] the import has to go
     with it, and then the spine's own root is the Native state -- which has a
     producer, so there is no root at all. *)
  let no_root =
    {
      valid with
      F.states =
        List.filteri (fun i _ -> i > 0) valid.F.states
        |> List.map (fun (s : F.State.t) ->
            if String.equal s.F.State.id "s/native/000" then
              { s with F.State.produced_by = None }
            else s);
      transitions = List.tl valid.F.transitions;
    }
  in
  check "no pt2 root" no_root;
  let two_roots =
    {
      valid with
      F.states = valid.F.states @ [ state "s/pt2/001" L.Pt2 "second" ];
    }
  in
  check "two pt2 roots" two_roots;
  [%expect
    {|
    no pt2 root                no pt2 root state
    two pt2 roots              2 root states, expected one |}]

let%expect_test "duplicate ids" =
  check "duplicate state"
    { valid with F.states = valid.F.states @ [ List.hd valid.F.states ] };
  check "duplicate transition"
    {
      valid with
      F.transitions = valid.F.transitions @ [ List.hd valid.F.transitions ];
    };
  [%expect
    {|
    duplicate state            duplicate flow state s/pt2/000
    duplicate transition       duplicate flow transition t/native/000 |}]

let%expect_test "an endpoint that names nothing" =
  let flow =
    {
      valid with
      F.transitions =
        valid.F.transitions
        @ [
            transition "t/native/009" "s/native/003" "s/native/404"
              F.Transition.Pack;
          ];
    }
  in
  check "dangling endpoint" flow;
  [%expect
    {| dangling endpoint          transition names unknown state s/native/404 |}]

(* --- the aggregates --- *)

let%expect_test "the two counts are checked before the walks they bound" =
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_states:4 ~max_transitions:4 limits)
  in
  Format.printf "states   %a@." pp_ok (F.validate ~limits:tight valid);
  let tight_t =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_states:64 ~max_transitions:4 limits)
  in
  Format.printf "trans    %a@." pp_ok (F.validate ~limits:tight_t valid);
  [%expect
    {|
    states   flow states = 8 is over the ceiling
    trans    flow transitions = 7 is over the ceiling |}]

(* --- the legal table --- *)

let%expect_test "every legal triple, enumerated" =
  List.iter
    (fun (b, k, a) ->
      Printf.printf "%-9s --%-14s-> %s\n" (L.to_string b) k (L.to_string a))
    F.legal_triples;
  [%expect
    {|
    pt2       --import        -> native
    native    --pass          -> native
    native    --pack          -> native
    native    --cross_dialect -> native4d
    native4d  --pass          -> native4d
    native4d  --pack          -> native4d
    native    --adapt         -> symbolic
    symbolic  --pass          -> symbolic
    symbolic  --adapt         -> kernel
    kernel    --pass          -> kernel |}]
