(* See pass.mli. *)

open Graph_ir

(* --- Dialect-independent, and therefore OUTSIDE the functor.

   [Pass.Make] is applied twice — once here as [Make (Native_side)], once in
   native4d/framework.ml as [Pass4] — and records declared inside a functor are
   distinct types in each application. So a counter declared inside could not
   merge a Native audit with a Native4D one, which is exactly what a summary
   over a whole pipeline has to do. [Map_verify] is the precedent in this very
   dependency: [Report], [Entry], [Outcome], [Verdict], [Coverage], [Tally] and
   [Policy] all sit outside [Make_pair], which is why [Audit.t] below can carry
   a [Map_verify.Report.t] at all. *)

module Frame = struct
  (* NESTED, not a path plus one scalar: pipeline.ml's [relayout] is
     [fixpoint (sequence "relayout" [ fixpoint chain_permute; ... ])], so a leaf
     runs under a stack of composites and an iteration number belongs to the
     composite it iterates, not to the leaf. *)
  type t = { name : string; iteration : int option }

  let pp fmt t =
    match t.iteration with
    | None -> Fmt.string fmt t.name
    | Some i -> Fmt.pf fmt "%s[%d]" t.name i
end

module Count_overflow = struct
  (* ONE payload naming the counter, rather than three near-identical error
     rows. Every counter here is [int64] and every increment is checked BEFORE
     the addition: a check on a wrapped sum is not a bound. *)
  type counter = Audit_reports | Outcome_bucket of string | Execution_index
  type t = { counter : counter }

  let pp_counter fmt = function
    | Audit_reports -> Fmt.string fmt "audit reports"
    | Outcome_bucket label -> Fmt.pf fmt "outcome bucket %S" label
    | Execution_index -> Fmt.string fmt "execution index"

  let pp fmt t = Fmt.pf fmt "@[<h>%a overflowed@]" pp_counter t.counter
end

type count_error = [ `Count_overflow of Count_overflow.t ]

let add_checked counter a b =
  (* [Int64.max_int - b] cannot underflow for [b >= 0], and every count here is
     non-negative by construction ([of_bindings] rejects the rest). *)
  if Int64.compare a (Int64.sub Int64.max_int b) > 0 then
    Err.fail (`Count_overflow { Count_overflow.counter })
  else Err.return (Int64.add a b)

module Outcome_counts = struct
  (* Counts by outcome label, sharing ONE vocabulary with [Map_verify.Tally]:
     both key on [Map_verify.Outcome.label], so a [Tested Agrees] cluster and a
     [Sampled n] proof each get their own bucket and neither is folded into
     "proved". [Tally] has [int] counts and no merge, which is why this exists
     beside it rather than extending it — and why [Map_verify] stays reused. *)

  (* An ORDERED association list, in the canonical order below, so encoding is
     deterministic on both backends. *)
  type t = (string * int64) list

  let empty = []
  let bindings t = t

  (* --- the label grammar ---

     Not a fixed list: [Tally] keys on [Verdict.label] optionally suffixed
     " [sampled n]" for a varying [n], so "[sampled 7]" and "[sampled 8]" are
     different labels. It is still checkable, because [Verdict.label] drops
     every payload and is therefore finite. *)

  let sampled_prefix = " [sampled "
  let sampled_suffix = "]"

  (* [1 <= n <= Int64.max_int]: a strict superset of BOTH backends' [int]
     domains, so a native producer holding a sampled count above 2^31 is inside
     it. Validated LEXICALLY and never converted — parsing through [int] would
     wrap under js_of_ocaml before any comparison, which is the defect the bound
     exists to prevent. *)
  let max_digits = 19
  let max_int64_digits = "9223372036854775807"

  let valid_digits d =
    let n = String.length d in
    n >= 1 && n <= max_digits
    (* No leading zero, which also rejects a bare "0" — the domain starts at 1.
       It is what makes (digit count, digit string) a NUMERIC order below. *)
    && d.[0] <> '0'
    && String.for_all (fun c -> c >= '0' && c <= '9') d
    (* At the maximum width, compare as text against [Int64.max_int]'s decimal.
       No conversion anywhere on this path. *)
    && (n < max_digits || String.compare d max_int64_digits <= 0)

  (* The longest label the grammar admits, computed rather than configured: it
     is an invariant of this module, not a policy knob. *)
  let max_label_bytes =
    List.fold_left
      (fun acc l -> max acc (String.length l))
      0 Map_verify.Verdict.labels
    + String.length sampled_prefix
    + String.length max_int64_digits
    + String.length sampled_suffix

  (* [Some (base, Some digits)] for a sampled label, [Some (base, None)] for a
     bare one, [None] when it is not in the grammar at all. *)
  let split label =
    let plen = String.length sampled_prefix in
    let slen = String.length sampled_suffix in
    let n = String.length label in
    (* A base verdict never contains '[' — they read "proved (structural)",
       "unproved (too large)" — so the first one, if any, is the prefix's. It
       sits one character into [sampled_prefix], which is where the base ends. *)
    match String.index_opt label '[' with
    | None ->
        if List.mem label Map_verify.Verdict.labels then Some (label, None)
        else None
    | Some bracket ->
        let base_len = bracket - 1 in
        if base_len < 0 || base_len + plen + slen > n then None
        else
          let base = String.sub label 0 base_len in
          let digits_len = n - base_len - plen - slen in
          if
            digits_len <= 0
            || String.sub label base_len plen <> sampled_prefix
            || String.sub label (n - slen) slen <> sampled_suffix
          then None
          else
            let digits = String.sub label (base_len + plen) digits_len in
            if List.mem base Map_verify.Verdict.labels && valid_digits digits
            then Some (base, Some digits)
            else None

  (* (index of the base verdict, digit count, digit string). Forbidding leading
     zeros is what makes the last two a NUMERIC order, so [sampled 9] sorts
     before [sampled 10]; jsoo never has to represent a large suffix as an
     [int]. *)
  let rank label =
    match split label with
    | None -> (max_int, 0, label)
    | Some (base, digits) ->
        let base_index =
          let rec find i = function
            | [] -> max_int
            | l :: rest -> if l = base then i else find (i + 1) rest
          in
          find 0 Map_verify.Verdict.labels
        in
        let d = Option.value digits ~default:"" in
        (base_index, String.length d, d)

  let compare_labels a b =
    let ia, na, da = rank a and ib, nb, db = rank b in
    match Int.compare ia ib with
    | 0 -> ( match Int.compare na nb with 0 -> String.compare da db | c -> c)
    | c -> c

  let canonical t = List.sort (fun (a, _) (b, _) -> compare_labels a b) t

  let bump label n t =
    let rec go = function
      | [] -> Err.return [ (label, n) ]
      | (l, m) :: rest when l = label ->
          let open Err.Syntax in
          let+ total = add_checked (Count_overflow.Outcome_bucket label) m n in
          (l, total) :: rest
      | binding :: rest ->
          let open Err.Syntax in
          let+ rest = go rest in
          binding :: rest
    in
    let open Err.Syntax in
    let+ t = go t in
    canonical t

  let add t outcome = bump (Map_verify.Outcome.label outcome) 1L t

  let of_report (report : Map_verify.Report.t) =
    let open Err.Syntax in
    Err.List.fold_left
      (fun acc (e : Map_verify.Entry.t) -> add acc e.outcome)
      empty report.entries
    |> fun r ->
    let+ t = r in
    t

  let merge a b =
    let open Err.Syntax in
    let+ t = Err.List.fold_left (fun acc (label, n) -> bump label n acc) a b in
    t

  module Invalid = struct
    type kind =
      | Negative_count
      | Duplicate_label
      | Malformed_label
      | Label_too_long

    type t = { label : string; kind : kind }

    let pp_kind fmt = function
      | Negative_count -> Fmt.string fmt "negative count"
      | Duplicate_label -> Fmt.string fmt "duplicate label"
      | Malformed_label -> Fmt.string fmt "not an outcome label"
      | Label_too_long -> Fmt.string fmt "label too long"
  end

  type invalid = [ `Invalid_counts of Invalid.t ]

  let pp_invalid fmt (`Invalid_counts { Invalid.label; kind }) =
    Fmt.pf fmt "@[<h>invalid outcome counts: %a (%S)@]" Invalid.pp_kind kind
      label

  (* THE way back in, and the reason [t] can be abstract while crossing a wire.
     Replaying [add] is not an inverse: the wire carries labels and counts, not
     [Map_verify.Outcome.t] values. It is also what lets a test seed a bucket at
     [Int64.max_int] — without which neither overflow check above could ever be
     shown to fire. *)
  let of_bindings bindings =
    let reject label kind =
      Err.fail (`Invalid_counts { Invalid.label; kind })
    in
    let rec check seen = function
      | [] -> Err.return ()
      | (label, n) :: rest ->
          if String.length label > max_label_bytes then
            reject label Invalid.Label_too_long
          else if split label = None then reject label Invalid.Malformed_label
          else if Int64.compare n 0L < 0 then
            reject label Invalid.Negative_count
          else if List.mem label seen then reject label Invalid.Duplicate_label
          else check (label :: seen) rest
    in
    let open Err.Syntax in
    let+ () = check [] bindings in
    canonical bindings

  (* [Jsont_bytesrw.decode_string] returns [('a, string) result], so a [Jsont.t]
     has no typed error channel and this row cannot propagate through one. The
     drop is deliberate and therefore NAMED, per CLAUDE.md — it is
     distinguishable from an accidental unwrap. OCaml callers and tests still
     get the typed row from [of_bindings]; only the DECODER reports a Jsont
     message. [Err.export] supplies the [Export] mark, which this helper
     previously omitted. *)
  let to_jsont r =
    match Err.export ~pos:__POS__ r with
    | Ok v -> v
    | Error k -> Jsont.Error.msg Jsont.Meta.none (Fmt.str "%a" pp_invalid k)

  module Binding = struct
    type t = { label : string; count : int64 }

    let jsont =
      Jsont.Object.map ~kind:"Outcome_count" (fun label count ->
          { label; count })
      |> Jsont.Object.mem "label" Jsont.string ~enc:(fun b -> b.label)
      (* [int64_as_string], never [Jsont.int64], which is adaptive
         number-or-string: a JS consumer must know which shape to expect, and a
         count above 2^53 has no exact [Number]. *)
      |> Jsont.Object.mem "count" Jsont.int64_as_string ~enc:(fun b -> b.count)
      |> Jsont.Object.finish
  end

  let jsont =
    (* An ARRAY, not an object: the canonical order is part of what makes the
       encoding deterministic across backends, and a JSON object keyed by label
       would be re-ordered by whatever map the decoder built it in. Decoding
       goes through [of_bindings], so a malformed binding cannot become an
       unchecked map. *)
    Jsont.map ~kind:"Outcome_counts"
      ~dec:(fun bs ->
        to_jsont
          (of_bindings
             (List.map (fun { Binding.label; count } -> (label, count)) bs)))
      ~enc:(fun t ->
        List.map (fun (label, count) -> { Binding.label; count }) (bindings t))
      (Jsont.list Binding.jsont)
end

module Exec_id = struct
  (* Authoritative PASS-EXECUTION provenance, and NOT a state identity: [N]
     changed steps produce [N + 1] states, and there is no execution at all for
     an import, a cross-dialect conversion, or a run in which no pass changed
     anything.

     UNIQUE WITHIN ONE TRACE SCOPE ONLY. [Make] is applied per dialect and each
     application seeds its ordinals from zero, while [per_node ~name] takes an
     arbitrary string — so a Native leaf and a Native4D leaf can hold EQUAL
     values. Hoisting this out of the functor made the two comparable; it did
     not make them distinct. Anything joining across dialects must qualify by
     scope. *)
  type t = { frames : Frame.t list; leaf : string; index : int64 }

  let pp fmt t =
    let sep fmt () = Fmt.string fmt "/" in
    if t.frames = [] then Fmt.pf fmt "%s#%Ld" t.leaf t.index
    else Fmt.pf fmt "%a/%s#%Ld" (Fmt.list ~sep Frame.pp) t.frames t.leaf t.index

  (* The ONLY way an ordinal advances. Checks the boundary BEFORE adding, so no
     caller is left with [Int64.add index 1L] and a look at the result. *)
  let next_ordinal index = add_checked Count_overflow.Execution_index index 1L
end

module Audit = struct
  (* What verifying one pass's step found. Only produced for an ACCEPTED report
     — a rejected one becomes a [`Verification] error instead.

     Keyed by [Exec_id.t] rather than by a bare pass name: a [fixpoint]
     contributes one audit per iteration and a [sequence] can hold the same leaf
     twice, so a name alone does not say which execution a report belongs to. *)
  type t = { id : Exec_id.t; report : Map_verify.Report.t }
end

module Audit_summary = struct
  (* At most ONE of these per log, holding what the retained reports could not.
     Rev. 5's "one Summary per overflowing leaf" kept the cardinality unbounded
     while only shrinking the payload. *)
  type t = { omitted_reports : int64; counts : Outcome_counts.t }

  let empty = { omitted_reports = 0L; counts = Outcome_counts.empty }

  (* RESULT-VALUED, and the row is [count_error], so the same row
     [Outcome_counts] raises flows out. [omitted_reports] and each outcome
     bucket are checked independently, so an overflow of either names itself. *)
  let add t report =
    let open Err.Syntax in
    let* omitted_reports =
      add_checked Count_overflow.Audit_reports t.omitted_reports 1L
    in
    let* counts = Outcome_counts.of_report report in
    let+ counts = Outcome_counts.merge t.counts counts in
    { omitted_reports; counts }
end

module Audit_log = struct
  (* GENUINELY bounded: at most [max_reports] cells, plus at most ONE aggregate.
     Retaining a summary per overflowing leaf would shrink the payload while
     leaving the CARDINALITY unbounded — one cell, one id and one tally per
     changed leaf — which is not a bound at all. *)
  type t = { reports : Audit.t list; overflow : Audit_summary.t option }

  let empty = { reports = []; overflow = None }

  let fold_away t (report : Map_verify.Report.t) =
    let open Err.Syntax in
    let+ summary =
      Audit_summary.add
        (Option.value t.overflow ~default:Audit_summary.empty)
        report
    in
    { t with overflow = Some summary }

  (* Enforced BEFORE retention, not by projecting an already-built log: a limit
     applied while projecting arrives after the memory is gone. *)
  let push ~max_reports t (audit : Audit.t) =
    if List.compare_length_with t.reports max_reports < 0 then
      Err.return { t with reports = t.reports @ [ audit ] }
    else fold_away t audit.report

  let concat ~max_reports a b =
    let open Err.Syntax in
    let* merged =
      Err.List.fold_left
        (fun acc audit -> push ~max_reports acc audit)
        a b.reports
    in
    match b.overflow with
    | None -> Err.return merged
    | Some s ->
        let base = Option.value merged.overflow ~default:Audit_summary.empty in
        let* omitted_reports =
          add_checked Count_overflow.Audit_reports base.omitted_reports
            s.omitted_reports
        in
        let+ counts = Outcome_counts.merge base.counts s.counts in
        {
          merged with
          overflow = Some { Audit_summary.omitted_reports; counts };
        }

  let retained t = List.length t.reports

  let omitted t =
    match t.overflow with
    | None -> 0L
    | Some s -> s.Audit_summary.omitted_reports
end

module Make (S : Side.S) = struct
  module View = Graph_view.Make (S.Dialect)
  module Rw = Rewrite.Make (S)
  module Rcp = Recipe.Make (S)
  module Pat = Pattern.Make (S.Dialect)
  module Rgn = Region.Make (S.Dialect)
  module Verify = Map_verify.Make_pair (S) (S)

  type node = S.op Graph_common.Node.t

  open Err.Syntax

  module Verification = struct
    (* Two distinct failures, and the caller needs to tell them apart: the
       verifier itself can error (a map that does not describe its two graphs, a
       missing signature), or it can succeed and the policy reject what it found.
       Both carry the pass name, since the point of verifying per step is to say
       WHICH rewrite is at fault. *)
    type problem = Error of Map_verify.error | Rejected of Map_verify.Report.t
    type t = { pass : string; problem : problem }

    let pp fmt t =
      match t.problem with
      | Error e ->
          Fmt.pf fmt "@[<h>pass %s: verifier failed: %a@]" t.pass
            Map_verify.pp_error e
      | Rejected report ->
          Fmt.pf fmt "@[<v 2>pass %s rejected: %s@,%a@]" t.pass
            (Map_verify.Report.summary report)
            Map_verify.Report.pp_verdicts report
  end

  (* The row WIDENS. It is closed, so without these two a [let*] over
     [Audit_summary.add] or [Exec_id.next_ordinal] inside [run] would not
     typecheck against the declared [error]. [Native_interp] already carries
     [`Transform of Pass.error], so widening here reaches the interpreter's
     result with no further change there. *)
  type error =
    [ Rw.error
    | `Not_converged of string
    | `Verification of Verification.t
    | count_error
    | `Malformed_outcome of string ]

  let pp_error ppf : [< error ] -> unit = function
    | #Rw.error as e -> Rw.pp_error ppf e
    | `Not_converged name -> Fmt.pf ppf "@[<h>pass %s did not converge@]" name
    | `Verification v -> Verification.pp ppf v
    | `Count_overflow c -> Count_overflow.pp ppf c
    | `Malformed_outcome name ->
        Fmt.pf ppf "@[<h>pass %s returned a malformed outcome@]" name

  (* What verification a run is under. Threaded THROUGH the pass tree rather than
     applied at the top, because a composite verified only at its boundary is
     barely verified at all: a [fixpoint] iteration or a [sequence] member can be
     wrong and be cancelled by a later one, and the error would name the composite
     rather than the pass that caused it. *)
  module Trace = struct
    module Entry = struct
      (* A NAMED, parameterised payload rather than an inline record. The
         objection to naming it was that the module would have to be
         parameterised by the ['v] the existential hides — it does, and that is
         the ordinary way to write this: ['v] is quantified at [t], so the
         existential is exactly as opaque as the inline form. Parameterising a
         payload so it can be declared beside the construct that quantifies over
         it is what CLAUDE.md's record-module rule asks for. *)
      module Payload = struct
        type 'v t = { id : Exec_id.t; before : 'v Rw.t; step : 'v Rw.step }
      end

      type t = Entry : 'v Payload.t -> t
    end

    (* [length entries <= max_trace_entries]. [truncated] so a reader is never
       told a partial trace is a whole one. *)
    type t = { entries : Entry.t list; truncated : bool }

    let empty = { entries = []; truncated = false }
    let length t = List.length t.entries

    (* Enforced BEFORE retention, for the same reason the audit budget is: a
       limit applied while projecting an already-built trace arrives after the
       memory is gone. Transformation CONTINUES either way — the final graph and
       the composed report stay available, and only the per-step detail is
       lost. *)
    let push ~max_entries t entry =
      if List.compare_length_with t.entries max_entries < 0 then
        { t with entries = t.entries @ [ entry ] }
      else { t with truncated = true }

    let concat ~max_entries a b =
      List.fold_left (push ~max_entries) a b.entries |> fun merged ->
      { merged with truncated = merged.truncated || b.truncated }
  end

  (* [frames] and [index] are the traversal position, threaded DOWN; the advance
     comes back up through [outcome.next_index], because [ctx] is immutable and
     [run_with] hands the same one to every pass. *)
  type ctx = {
    budget : Map_verify.Budget.t option;
    policy : Map_verify.Policy.t option;
    probe : int option;
    trace : bool;
    max_trace_entries : int;
    max_audit_reports : int;
    frames : Frame.t list;
    index : int64;
  }

  let default_max_audit_reports = 256
  let default_max_trace_entries = 1024

  let no_verification =
    {
      budget = None;
      policy = None;
      probe = None;
      trace = false;
      max_trace_entries = default_max_trace_entries;
      max_audit_reports = default_max_audit_reports;
      frames = [];
      index = 0L;
    }

  (* A step plus what verifying it found. Carried through the tree rather than
     handed to a callback, so nothing here needs mutable state to observe its own
     traversal; [run_all] drops the audits for the callers that do not want them.

     [next_index] is the ordinal AFTER this pass. A leaf that changed something
     consumes one; a converged identity sweep consumes none, which is what keeps
     ordinals dense over the executions that actually happened. *)
  type 'v outcome = {
    audits : Audit_log.t;
    trace : Trace.t;
    next_index : int64;
    step : 'v Rw.step;
  }

  type t = {
    name : string;
    run : 'v. ctx -> 'v Rw.t -> ('v outcome, error) Err.t;
  }

  type env = {
    constant_store : Constant_store.t;
    constants : Tensor.packed Tensor_id.Map.t;
    view : View.t;
  }

  type per_node = { on_node : 'v. env -> node -> ('v, unit) Rcp.t option }
  type 'a builder = { build : 'v. 'a -> Rgn.t -> ('v, unit) Rcp.t }

  (* The driver's own rank-2 hook: [of_sweep] runs [collect] at the version of the
     state it is planning against. *)
  type collector = { collect : 'v. env -> ('v, unit) Rcp.t list }

  let lift r = (r :> ('a, error) Err.t)

  (* One sweep: collect the builders a pass offers, plan them one after another so
     their allocations are contiguous, merge, and apply once. Merging is what makes
     the sweep a single step with a single mapping, rather than N steps whose maps
     the caller would have to compose. *)
  let sweep state builders =
    let* planned, _ =
      List.fold_left
        (fun acc builder ->
          let* recipes, alloc = acc in
          let+ recipe, alloc = lift (Rw.plan state alloc builder) in
          (recipe :: recipes, alloc))
        (Err.return ([], Rw.allocator state))
        builders
    in
    match List.rev planned with
    | [] -> Err.return None
    | first :: rest ->
        let* merged =
          List.fold_left
            (fun acc recipe ->
              let* acc = acc in
              lift (Rw.merge acc recipe))
            (Err.return first) rest
        in
        let+ step = lift (Rw.apply state merged) in
        Some step

  let identity_step state = Rw.Step (state, Graph_map.identity)

  (* Convergence is "the graph stopped changing", read off the map being empty —
     a step that rewrote nothing produces the identity map, which is exactly the
     signal. Declared here because both [of_sweep] and [fixpoint] consult it. *)
  let changed (map : ('a, 'b) Graph_map.t) =
    not
      (Correspondence.is_empty (Graph_map.values map)
      && Node_map.is_empty (Graph_map.nodes map))

  (* Verify one step against the state it came from, naming the execution. Only
     ever reached for a CHANGED step: an identity map has nothing to check, and
     on a real graph checking every cluster of a no-op sweep is the dominant
     cost. *)
  let verified id ctx state step =
    let name = id.Exec_id.leaf in
    match ctx.policy with
    | None -> Err.return Audit_log.empty
    | Some policy -> (
        (* [Map_verify.step] is Native-only — it reaches into [Rewrite] — so its
           two lines are inlined here, where both modules are in scope at this
           dialect. Constants are read from each state SEPARATELY: a fold
           consumes and deletes one, so it exists only in the before-state. *)
        let (Rw.Step (after, map)) = step in
        match
          Verify.run ?budget:ctx.budget ?probe:ctx.probe map
            ~src:(Rw.snapshot state) ~src_constants:(Rw.constants state)
            ~src_constant_store:(Rw.constant_store state)
            ~dst:(Rw.snapshot after) ~dst_constants:(Rw.constants after)
            ~dst_constant_store:(Rw.constant_store after)
        with
        | Error e ->
            Err.fail
              (`Verification
                 {
                   Verification.pass = name;
                   problem = Verification.Error (Err.Error.kind e);
                 })
        | Ok report ->
            if Map_verify.Policy.accepts policy report then
              Audit_log.push ~max_reports:ctx.max_audit_reports Audit_log.empty
                { Audit.id; report }
            else
              Err.fail
                (`Verification
                   {
                     Verification.pass = name;
                     problem = Verification.Rejected report;
                   }))

  (* The TRUSTED leaf constructor, and therefore where the ordinal rule lives.
     [run_with] receives a [t] as [{name; run}] and cannot tell a leaf from a
     [sequence] from a [fixpoint] — a composite legitimately advances by many
     leaves — so it cannot enforce "at most one ordinal per leaf". Here it holds
     by construction. *)
  let of_sweep ~name { collect } =
    {
      name;
      run =
        (fun ctx state ->
          let env =
            {
              constant_store = Rw.constant_store state;
              constants = Rw.constants state;
              view = Rw.view state;
            }
          in
          let* result = sweep state (collect env) in
          let step =
            match result with None -> identity_step state | Some step -> step
          in
          let (Rw.Step (_, map)) = step in
          if not (changed map) then
            (* A converged identity sweep consumes NO ordinal, has nothing to
               verify and is not traced: it is not an execution. That is what
               keeps ordinals dense over the executions that really happened
               rather than over the attempts. *)
            Err.return
              {
                audits = Audit_log.empty;
                trace = Trace.empty;
                next_index = ctx.index;
                step;
              }
          else
            let id =
              { Exec_id.frames = ctx.frames; leaf = name; index = ctx.index }
            in
            let* next_index = Exec_id.next_ordinal ctx.index in
            let+ audits = verified id ctx state step in
            let trace =
              if not ctx.trace then Trace.empty
              else
                Trace.push ~max_entries:ctx.max_trace_entries Trace.empty
                  (Trace.Entry.Entry
                     { Trace.Entry.Payload.id; before = state; step })
            in
            { audits; trace; next_index; step });
    }

  let per_node ~name { on_node } =
    of_sweep ~name
      {
        collect =
          (fun env ->
            List.filter_map (on_node env)
              (Graph_common.nodes (View.graph env.view)));
      }

  let of_pattern ~name ~pattern ~build:{ build } =
    of_sweep ~name
      {
        collect =
          (fun env ->
            Pat.scan pattern env.view
            |> List.map (fun (value, region) -> build value region));
      }

  (* The accumulated mapping's destination changes every iteration, so it cannot
     be carried as a [('v,'w) Graph_map.t] with 'w fixed. [Rw.step] already
     packages a state together with the map reaching it, existentially — so the
     accumulator IS a step, and composing into it keeps the caller's view as one
     mapping from the state it handed in to the final graph.

     The frame carries the ITERATION, refreshed each round, so a leaf running
     under three rounds of a fixpoint produces three distinguishable ids. *)
  let fixpoint ?(max_iters = 16) inner =
    {
      name = inner.name;
      run =
        (fun ctx state ->
          let rec go : type v.
              int ->
              int ->
              Audit_log.t * Trace.t ->
              int64 ->
              v Rw.step ->
              (v outcome, error) Err.t =
           fun fuel iteration (audits, trace) index (Rw.Step (state, acc)) ->
            if fuel <= 0 then Err.fail (`Not_converged inner.name)
            else
              (* [ctx], so EVERY iteration is verified rather than only the
                 composite the caller sees. *)
              let inner_ctx =
                {
                  ctx with
                  frames =
                    ctx.frames
                    @ [
                        { Frame.name = inner.name; iteration = Some iteration };
                      ];
                  index;
                }
              in
              let* inner_out = inner.run inner_ctx state in
              let (Rw.Step (next, map)) = inner_out.step in
              let* audits =
                Audit_log.concat ~max_reports:ctx.max_audit_reports audits
                  inner_out.audits
              in
              let trace =
                Trace.concat ~max_entries:ctx.max_trace_entries trace
                  inner_out.trace
              in
              if changed map then
                go (fuel - 1) (iteration + 1) (audits, trace)
                  inner_out.next_index
                  (Rw.Step (next, Graph_map.compose acc map))
              else
                Err.return
                  {
                    audits;
                    trace;
                    next_index = inner_out.next_index;
                    step = Rw.Step (state, acc);
                  }
          in
          go max_iters 0
            (Audit_log.empty, Trace.empty)
            ctx.index (identity_step state));
    }

  (* Every pass verifies its own step, so this only threads the context and
     concatenates what came back. *)
  (* Every pass verifies its own step, so this threads the context, THREADS THE
     ORDINAL between members, and concatenates what came back. *)
  let run_with ctx state passes =
    let rec go : type v. int64 -> v Rw.t -> t list -> (v outcome, error) Err.t =
     fun index state passes ->
      match passes with
      | [] ->
          Err.return
            {
              audits = Audit_log.empty;
              trace = Trace.empty;
              next_index = index;
              step = identity_step state;
            }
      | pass :: rest ->
          let* first = pass.run { ctx with index } state in
          let (Rw.Step (next, map)) = first.step in
          (* STRUCTURAL SANITY ONLY. [run_with] cannot prove intent — a
             hand-built pass can fabricate a plausible id — so it checks what it
             can see: the ordinal must not go backwards. The per-leaf and
             exact-frame rules live in [of_sweep], which is private. *)
          let* () =
            if Int64.compare first.next_index index < 0 then
              Err.fail (`Malformed_outcome pass.name)
            else Err.return ()
          in
          let* rest = go first.next_index next rest in
          let (Rw.Step (final, rest_map)) = rest.step in
          let+ audits =
            Audit_log.concat ~max_reports:ctx.max_audit_reports first.audits
              rest.audits
          in
          {
            audits;
            trace =
              Trace.concat ~max_entries:ctx.max_trace_entries first.trace
                rest.trace;
            next_index = rest.next_index;
            step = Rw.Step (final, Graph_map.compose map rest_map);
          }
    in
    go ctx.index state passes

  let run_reporting ?verify ?verify_budget ?verify_probe ?trace
      ?max_trace_entries ?max_audit_reports state passes =
    run_with
      {
        no_verification with
        budget = verify_budget;
        policy = verify;
        probe = verify_probe;
        trace = Option.value trace ~default:false;
        max_trace_entries =
          Option.value max_trace_entries ~default:default_max_trace_entries;
        max_audit_reports =
          Option.value max_audit_reports ~default:default_max_audit_reports;
      }
      state passes

  let run_all ?verify ?verify_budget ?verify_probe state passes =
    let+ out =
      run_reporting ?verify ?verify_budget ?verify_probe state passes
    in
    out.step

  (* A fixed list of passes as one named pass, so a caller can [fixpoint] the
     whole group — needed when the passes unlock each other and no single
     non-interleaved round through them all is enough. [ctx] is forwarded, so a
     member is verified as it runs rather than only through the composite map the
     sequence hands back. *)
  let sequence ~name passes =
    {
      name;
      run =
        (fun ctx state ->
          (* The frame carries NO iteration: a sequence runs its members once,
             and an iteration number here would be a fixpoint's, one level
             up. *)
          run_with
            {
              ctx with
              frames = ctx.frames @ [ { Frame.name; iteration = None } ];
            }
            state passes);
    }
end

include Make (Native_side)
