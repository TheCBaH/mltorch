(* Execution provenance: [Frame], [Exec_id], the ordinal rule, and the bounded
   [Audit_log].

   The shape under test is the real one. [Pipeline.relayout] is
   [fixpoint (sequence "relayout" [ fixpoint chain_permute; ...; fixpoint
   sink_permute; ...; fixpoint sink_permute; ... ])] — a fixpoint inside a
   sequence inside a fixpoint, with a LEAF APPEARING TWICE. That last part is
   why an audit is keyed by [Exec_id.t] and not by a pass name: before this,
   two of resnet18's audits were both called "sink_permute" and nothing
   distinguished them.

   What the ordinal rule has to deliver:

     a leaf that CHANGED something consumes exactly one ordinal
     a converged identity sweep consumes NONE          (density)
     ordinals do not repeat within one run             (uniqueness)
     verification does not move them                   (independence) *)

(* A graph with a dead branch, so a [Dce] sweep has something to do on its first
   round and nothing on its second — which is what makes the identity-sweep case
   reachable without a hand-built pass. *)
let dead_branch () =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"dead_branch" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4) () in
      let* live = relu ~name:"live" x in
      let* _dead = relu ~name:"dead" x in
      relu ~name:"out" live)

(* [Rewrite.origin] hands back an EXISTENTIAL state, and [Pass.outcome] is
   parameterised by that tag, so an outcome cannot leave the scope it was
   produced in. Everything these tests assert on — the ordinal, the retained and
   omitted counts, the ids — is monomorphic, so the summary is extracted inside
   the scope and the step is simply dropped. *)
module Summary = struct
  type t = {
    next_index : int64;
    retained : int;
    omitted : int64;
    traced : int;
    truncated : bool;
    ids : string list;
    trace_ids : string list;
  }

  let of_outcome (out : 'a Pass.outcome) =
    {
      next_index = out.next_index;
      retained = Pass.Audit_log.retained out.audits;
      omitted = Pass.Audit_log.omitted out.audits;
      traced = Pass.Trace.length out.trace;
      truncated = out.trace.truncated;
      ids =
        List.map
          (fun (a : Pass.Audit.t) -> Core.Pretty.to_string Pass.Exec_id.pp a.id)
          out.audits.reports;
      trace_ids =
        List.map
          (fun (Pass.Trace.Entry.Entry p) ->
            Core.Pretty.to_string Pass.Exec_id.pp p.id)
          out.trace.entries;
    }

  let pp ppf t =
    Fmt.pf ppf
      "@[<v>next_index=%Ld retained=%d omitted=%Ld traced=%d truncated=%b@,\
       audits: %a@,\
       trace:  %a@]"
      t.next_index t.retained t.omitted t.traced t.truncated
      (Fmt.list ~sep:Fmt.comma Fmt.string)
      t.ids
      (Fmt.list ~sep:Fmt.comma Fmt.string)
      t.trace_ids
end

type error = [ Pass.error | `Origin of Rewrite.error ]

let pp_error ppf : [< error ] -> unit = function
  | `Origin e -> Rewrite.pp_error ppf e
  | #Pass.error as e -> Pass.pp_error ppf e

let pp_result = Core.Pretty.err_result ~ok:Summary.pp ~error:pp_error

let summarize ?verify ?trace ?max_trace_entries ?max_audit_reports
    ?max_verified_steps ?graph passes =
  let open Err.Syntax in
  let graph = Option.value graph ~default:(dead_branch ()) in
  let* (Rewrite.Origin state) =
    Rewrite.origin graph |> Err.map_error (fun e -> `Origin e)
  in
  let+ out =
    Pass.run_reporting ?verify ?trace ?max_trace_entries ?max_audit_reports
      ?max_verified_steps state passes
    |> Err.map_error (fun e -> (e :> error))
  in
  Summary.of_outcome out

let run ?verify ?trace ?max_trace_entries ?max_audit_reports ?max_verified_steps
    ?graph passes =
  Format.printf "%a@." pp_result
    (summarize ?verify ?trace ?max_trace_entries ?max_audit_reports
       ?max_verified_steps ?graph passes)

(* --- the real nesting shape --- *)

let nested =
  (* fixpoint (sequence [ fixpoint leaf; fixpoint leaf ]) — the same leaf twice,
     as in [Pipeline.relayout]. *)
  Pass.fixpoint
    (Pass.sequence ~name:"group"
       [ Pass.fixpoint Dce.pass; Pass.fixpoint Dce.pass ])

let%expect_test "frames nest, and the same leaf twice is still distinguishable"
    =
  run ~verify:Map_verify.Policy.Reject_refuted [ nested ];
  [%expect
    {|
    next_index=1 retained=1 omitted=0 traced=0 truncated=false
    audits: group[0]/group/dce[0]/dce#0
    trace:
    |}]

let%expect_test "ordinals are dense: an identity sweep consumes none" =
  (* Two [Dce] passes in a row. The first removes the dead branch and consumes
     ordinal 0; the second finds nothing, produces an identity step, and
     consumes NOTHING — so [next_index] is 1, not 2. Density over executions
     that happened, rather than over attempts. *)
  run ~verify:Map_verify.Policy.Reject_refuted [ Dce.pass; Dce.pass ];
  [%expect
    {|
    next_index=1 retained=1 omitted=0 traced=0 truncated=false
    audits: dce#0
    trace:
    |}]

let%expect_test "verification does not move the ordinals" =
  (* [~verify] decides whether a REPORT is produced, never whether an execution
     happened. So the ordinal stream is identical with and without it — and
     without it there are no audits to retain at all. *)
  let next r = Result.map (fun (s : Summary.t) -> s.next_index) r in
  Format.printf "same_next_index=%b@."
    (next (summarize ~verify:Map_verify.Policy.Reject_refuted [ nested ])
    = next (summarize [ nested ]));
  run [ nested ];
  [%expect
    {|
    same_next_index=true
    next_index=1 retained=0 omitted=0 traced=0 truncated=false
    audits:
    trace:
    |}]

(* --- the bounded log --- *)

let%expect_test "the audit log is bounded in CARDINALITY, not just in payload" =
  (* Driven at [Audit_log] rather than through a pipeline. Reaching the budget
     from a pass list needs N executions that each CHANGE something, and the
     obvious fixture — N copies of one pass — gives exactly one: the first
     changes the graph and the rest converge. A version of this test built that
     way reports retained=0 and passes without ever touching the bound, which is
     the vacuity this suite exists to avoid.

     Twelve audits against a budget of three: three cells retained, nine folded
     into ONE aggregate. Retaining a summary per overflowing leaf would have
     kept twelve cells and called it bounded. *)
  let report verdict : Map_verify.Report.t =
    {
      entries =
        [
          {
            cluster =
              {
                src = Graph_ir.Tensor_id.Set.empty;
                dst = Graph_ir.Tensor_id.Set.empty;
                label = Correspondence.Identical;
              };
            group = [];
            outcome = { coverage = Map_verify.Coverage.not_applicable; verdict };
          };
        ];
    }
  in
  let audit i : Pass.Audit.t =
    {
      id = { frames = []; leaf = "leaf"; index = Int64.of_int i };
      report = report Map_verify.Verdict.Vacuous;
    }
  in
  let log =
    Err.List.fold_left
      (fun acc i -> Pass.Audit_log.push ~max_reports:3 acc (audit i))
      Pass.Audit_log.empty (List.init 12 Fun.id)
  in
  let pp_log ppf (l : Pass.Audit_log.t) =
    Fmt.pf ppf "retained=%d omitted=%Ld@,%a"
      (Pass.Audit_log.retained l)
      (Pass.Audit_log.omitted l)
      (Fmt.option ~none:(Fmt.any "no aggregate") (fun ppf s ->
           Fmt.pf ppf "aggregate: %a"
             (Fmt.list (fun ppf (lbl, n) -> Fmt.pf ppf "%s=%Ld" lbl n))
             (Pass.Outcome_counts.bindings s.Pass.Audit_summary.counts)))
      l.overflow
  in
  Format.printf "@[<v>%a@]@."
    (Core.Pretty.err_result ~ok:pp_log ~error:(fun ppf (`Count_overflow c) ->
         Pass.Count_overflow.pp ppf c))
    log;
  [%expect {|
    retained=3 omitted=9
    aggregate: vacuous=9
    |}]

let%expect_test "an exhausted verification-work budget skips audits visibly" =
  (* Unlike [max_audit_reports], this limit applies BEFORE calling the verifier.
     The changed pass and its trace still exist, while the omitted audit has no
     made-up outcome counts. *)
  run ~verify:Map_verify.Policy.Reject_refuted ~max_verified_steps:0 ~trace:true
    [ nested ];
  [%expect
    {|
    next_index=1 retained=0 omitted=1 traced=1 truncated=false
    audits:
    trace:  group[0]/group/dce[0]/dce#0
    |}]

(* --- the trace --- *)

let%expect_test "the trace records one entry per execution, under the same ids"
    =
  (* The trace and the audits index the SAME executions, so a reader can join
     "what this pass did" to "what verifying it found" without either side
     inventing an identity. Off by default, because retaining a state per
     execution is the expensive product here. *)
  run ~verify:Map_verify.Policy.Reject_refuted ~trace:true [ nested ];
  [%expect
    {|
    next_index=1 retained=1 omitted=0 traced=1 truncated=false
    audits: group[0]/group/dce[0]/dce#0
    trace:  group[0]/group/dce[0]/dce#0
    |}]

let%expect_test "trace and verification are independent products" =
  (* Tracing without verifying produces entries and no audits — they are driven
     by different flags over the same executions, so neither can be inferred
     from the other's presence. *)
  run ~trace:true [ nested ];
  [%expect
    {|
    next_index=1 retained=0 omitted=0 traced=1 truncated=false
    audits:
    trace:  group[0]/group/dce[0]/dce#0
    |}]

let%expect_test "an exhausted trace budget truncates and says so" =
  (* Budget zero, one execution: nothing retained and [truncated] set, so a
     reader is never handed a partial trace that looks whole. The run still
     COMPLETES — the graph and the audits survive, and only the per-step detail
     is lost. *)
  run ~verify:Map_verify.Policy.Reject_refuted ~trace:true ~max_trace_entries:0
    [ nested ];
  [%expect
    {|
    next_index=1 retained=1 omitted=0 traced=0 truncated=true
    audits: group[0]/group/dce[0]/dce#0
    trace:
    |}]

(* --- the checked ordinal --- *)

let%expect_test "next_ordinal reports at the boundary rather than wrapping" =
  (* [int64] on both backends, and the check runs BEFORE the addition — a look
     at [Int64.add index 1L] afterwards would see a wrapped negative and read as
     success. *)
  let show n =
    Format.printf "%Ld -> %a@." n
      (Core.Pretty.err_result
         ~ok:(fun ppf v -> Fmt.pf ppf "%Ld" v)
         ~error:(fun ppf (`Count_overflow c) -> Pass.Count_overflow.pp ppf c))
      (Pass.Exec_id.next_ordinal n)
  in
  show 0L;
  show 2147483647L;
  show (Int64.sub Int64.max_int 1L);
  show Int64.max_int;
  [%expect
    {|
    0 -> 1
    2147483647 -> 2147483648
    9223372036854775806 -> 9223372036854775807
    9223372036854775807 -> execution index overflowed
    |}]

(* --- what run_with can and cannot prove --- *)

let%expect_test "a pass returning a backwards ordinal is malformed" =
  (* [run_with] sees a [t] as [{name; run}] and cannot tell a leaf from a
     composite — a composite legitimately advances by many leaves, and a
     hand-built pass can fabricate a plausible prefix. So it checks only what it
     can see, and says [`Malformed_outcome] rather than claiming intent. *)
  let backwards =
    {
      Pass.name = "backwards";
      run =
        (fun _ctx state ->
          Err.return
            {
              Pass.audits = Pass.Audit_log.empty;
              trace = Pass.Trace.empty;
              next_index = -1L;
              step = Rewrite.Step (state, Graph_map.identity);
            });
    }
  in
  run [ backwards ];
  [%expect {| pass backwards returned a malformed outcome |}]

let%expect_test "a well-formed composite is accepted" =
  (* The control for the case above: nested composites advance the ordinal by
     many leaves and must NOT trip the structural check. *)
  run ~verify:Map_verify.Policy.Reject_refuted
    [ Pass.sequence ~name:"outer" [ nested; Dce.pass ] ];
  [%expect
    {|
    next_index=1 retained=1 omitted=0 traced=0 truncated=false
    audits: outer/group[0]/group/dce[0]/dce#0
    trace:
    |}]
