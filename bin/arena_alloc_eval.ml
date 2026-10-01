(* Allocator evaluation over the tracked model.json corpus: for every model and
   each normalized dialect, the arena's allocation problems (one per element
   kind), every existing strategy on them, the order search from each at every
   budget and seed, the production portfolio, and the reference minimum. No
   weights, no inputs, no tensor operation: placement quality and planning
   cost only.

   Output is versioned JSONL, one row per fact, flushed per model and in a
   deterministic order; [--summary] renders it as markdown, from the file, so a
   resumed run summarizes the same way as a fresh one. Exits nonzero on a
   corpus-count mismatch, a failed extraction, an invalid placement or an
   internal error; a classified Native4D refusal and a bounded reference search
   running out of work are results, not failures.

   See .ai/ (the tensor arena design) for what the figures mean. *)

module X = Arena_eval_extract
module J = Arena_eval_json
module IA = Interval_alloc
open Core.Storage_units

(* Reports are raw [int64] JSON: sizes leave their type here. *)
let b64 = Byte_size.to_int64
let version = 1

module Run = struct
  type t = {
    models_dir : string;
    expected_models : int;
    dialects : X.Dialect.t list;
    eval : Arena_eval.Config.t;
    output : string;
    summary : string option;
    models_output : string option;
    artifacts : string option;
    resume : bool;
  }
end

(* --- the corpus ---------------------------------------------------------- *)

module Entry = struct
  type t = { model : string; path : string; md5 : string }
end

(* A command's stdout, or [None] if it could not run or failed. *)
let command_output prog args =
  let out = Filename.temp_file "arena_alloc_eval" ".out" in
  Fun.protect
    ~finally:(fun () -> Sys.remove out)
    (fun () ->
      let cmd =
        Filename.quote_command prog args ~stdout:out ~stderr:"/dev/null"
      in
      if Sys.command cmd = 0 then
        Some (In_channel.with_open_bin out In_channel.input_all)
      else None)

let lines s = String.split_on_char '\n' s |> List.filter (( <> ) "")

(* The pinned producer's tracked [<model>/models/model.json] files when the
   directory is in a git work tree, otherwise a directory scan (a fixture tree
   is not one). *)
let manifest models_dir =
  let model_of path =
    match String.split_on_char '/' path with
    | [ model; "models"; "model.json" ] -> Some model
    | _ -> None
  in
  let source, models =
    match
      command_output "git"
        [ "-C"; models_dir; "ls-files"; "--"; "*/models/model.json" ]
    with
    | Some out when lines out <> [] ->
        ("git", List.filter_map model_of (lines out))
    | _ ->
        ( "directory",
          Sys.readdir models_dir |> Array.to_list
          |> List.filter (fun m ->
              Sys.file_exists
                (Filename.concat models_dir
                   (Filename.concat m "models/model.json"))) )
  in
  let models = List.sort String.compare models in
  let entries =
    List.map
      (fun model ->
        let path =
          Filename.concat models_dir (Filename.concat model "models/model.json")
        in
        let md5 =
          if Sys.file_exists path then Digest.to_hex (Digest.file path) else ""
        in
        { Entry.model; path; md5 })
      models
  in
  (source, entries)

let validate_manifest ~expected entries =
  let models = List.map (fun (e : Entry.t) -> e.model) entries in
  let rec dup = function
    | a :: (b :: _ as rest) -> if String.equal a b then Some a else dup rest
    | _ -> None
  in
  match
    ( dup models,
      List.find_opt (fun (e : Entry.t) -> String.equal e.md5 "") entries )
  with
  | Some m, _ -> Error (Fmt.str "duplicate model %s" m)
  | None, Some e -> Error (Fmt.str "missing %s" e.path)
  | None, None ->
      if List.length entries <> expected then
        Error
          (Fmt.str "expected %d models, found %d" expected (List.length entries))
      else Ok ()

(* --- per kind ------------------------------------------------------------- *)

module Problem_stats = struct
  type t = {
    blocks : int;
    positive : int;
    conflicts : int;
    peak_live_blocks : int;
    size_min : int64;
    size_max : int64;
    size_total : int64;
    lifetime_max : int;
    lifetime_mean : float;
  }
end

let problem_stats script =
  let events = IA.Script.events script in
  let length = List.length events in
  let ranges = Hashtbl.create 64 and order = ref [] in
  let live = ref 0 and peak = ref 0 in
  List.iteri
    (fun pos -> function
      | IA.Event.Alloc { key; size; _ } ->
          let k = Tensor_id.to_int key in
          Hashtbl.replace ranges k (pos, length, b64 size);
          order := k :: !order;
          incr live;
          peak := max !peak !live
      | IA.Event.Free key ->
          let k = Tensor_id.to_int key in
          let a, _, size = Hashtbl.find ranges k in
          Hashtbl.replace ranges k (a, pos, size);
          decr live)
    events;
  let blocks = List.rev_map (Hashtbl.find ranges) !order |> Array.of_list in
  let positive =
    Array.of_list
      (List.filter (fun (_, _, s) -> s <> 0L) (Array.to_list blocks))
  in
  let conflicts = ref 0 in
  Array.iteri
    (fun i (a, f, _) ->
      for j = i + 1 to Array.length positive - 1 do
        let a', f', _ = positive.(j) in
        if a < f' && a' < f then incr conflicts
      done)
    positive;
  let n = Array.length blocks in
  let fold f init = Array.fold_left f init blocks in
  {
    Problem_stats.blocks = n;
    positive = Array.length positive;
    conflicts = !conflicts;
    peak_live_blocks = !peak;
    size_min =
      (if n = 0 then 0L
       else fold (fun m (_, _, s) -> Int64.min m s) Int64.max_int);
    size_max = fold (fun m (_, _, s) -> Int64.max m s) 0L;
    size_total = fold (fun m (_, _, s) -> Int64.add m s) 0L;
    lifetime_max = fold (fun m (a, f, _) -> max m (f - a)) 0;
    lifetime_mean =
      (if n = 0 then 0.
       else
         float_of_int (fold (fun m (a, f, _) -> m + (f - a)) 0)
         /. float_of_int n);
  }

module Kind_result = struct
  type t = {
    kind : Alloc_script.Kind.t;
    digest : string;
    live_bound : int64;
    stats : Problem_stats.t;
    reference : Arena_eval.Reference_row.t;
    rows : Arena_eval.Row.t list;
    placed : int64;
        (** The production portfolio on the exact-size script: the pool a plan
            actually places, where every other figure is of the padded one. *)
  }
end

let pp_eval_error ppf = function
  | `Arena_placement e ->
      let id =
        match e with
        | `Duplicate_placement id
        | `Live_overflow id
        | `Offset_overflow id
        | `Pool_overflow id
        | `Unknown_key id
        | `Unplaced id ->
            id
        | `Misaligned { IA.Misaligned.block; _ }
        | `Out_of_pool { IA.Out_of_pool.block; _ } ->
            block.IA.Block.key
        | `Overlap { IA.Overlap.first; _ } -> first.IA.Block.key
      in
      Fmt.pf ppf "invalid placement at %a" Tensor_id.pp id
  | `Invalid_candidate { IA.Reference.Invalid_candidate.pool; lower; ceiling }
    ->
      Fmt.pf ppf "reference candidate %a outside [%a, %a]" Byte_size.pp pool
        Byte_size.pp lower Byte_size.pp ceiling

let problem_digest script =
  Digest.to_hex
    (Digest.string
       (String.concat ";"
          (List.map
             (function
               | IA.Event.Alloc { key; size; alignment } ->
                   Printf.sprintf "+%d:%Ld@%Ld" (Tensor_id.to_int key)
                     (b64 size)
                     (Byte_alignment.to_int64 alignment)
               | IA.Event.Free key ->
                   Printf.sprintf "-%d" (Tensor_id.to_int key))
             (IA.Script.events script))))

(* The minimum is of the padded script, where it is provable; [placed] is the
   production planner on the exact one, what a plan holds. *)
let evaluate_kind ~now (run : Run.t)
    ({ Arena_problem.Kind_problem.kind; padded = script; _ } as problem) =
  let open Err.Syntax in
  let* live_bound =
    IA.lower_bound script
    |> Err.map_error ~pos:__POS__ (fun e ->
        `Arena_placement (e :> Arena_plan.Placement_error.t))
  in
  let* rows = Arena_eval.strategies ~now run.eval script in
  let* reference = Arena_eval.reference ~now run.eval script in
  let+ placed = Arena_eval.placed run.eval problem in
  {
    Kind_result.kind;
    digest = problem_digest script;
    live_bound = b64 live_bound;
    stats = problem_stats script;
    reference;
    rows;
    placed = b64 placed;
  }

(* --- rows ---------------------------------------------------------------- *)

let kind_name k = Fmt.str "%a" Alloc_script.Kind.pp k

let status_name = function
  | IA.Reference.Status.Incomplete -> "incomplete"
  | Optimal_above_live_bound -> "optimal_above_live_bound"
  | Optimal_live_bound -> "optimal_live_bound"

let stop_name = function
  | IA.Reference.Stop.Closed -> "closed"
  | Depth_limit -> "depth_limit"
  | State_limit -> "state_limit"

let family = function
  | Arena_eval.Method.Constructive _ -> "constructive"
  | Improved _ -> "improved"
  | Portfolio -> "portfolio"

let strategy_name = function
  | Arena_eval.Method.Constructive s | Improved s ->
      Fmt.str "%a" IA.Strategy.pp s
  | Portfolio -> "portfolio"

let label (r : Arena_eval.Row.t) =
  match r.budget with
  | None -> Fmt.str "%a" Arena_eval.Method.pp r.method_
  | Some (i, s) -> Fmt.str "%a@%Ld/%Ld" Arena_eval.Method.pp r.method_ i s

let problem_row ids (k : Kind_result.t) =
  let s = k.stats in
  J.obj
    (ids
    @ [
        ("type", J.str "problem");
        ("kind", J.str (kind_name k.kind));
        ("problem_digest", J.str k.digest);
        ("blocks", J.int s.blocks);
        ("positive_blocks", J.int s.positive);
        ("conflicts", J.int s.conflicts);
        ("peak_live_blocks", J.int s.peak_live_blocks);
        ("size_min", J.i64 s.size_min);
        ("size_max", J.i64 s.size_max);
        ("size_total", J.i64 s.size_total);
        ("lifetime_max", J.int s.lifetime_max);
        ("lifetime_mean", J.num s.lifetime_mean);
        ("live_bound", J.i64 k.live_bound);
      ])

let reference_row ids (k : Kind_result.t) =
  let r = k.reference and b = k.reference.bounds in
  J.obj
    (ids
    @ [
        ("type", J.str "reference");
        ("kind", J.str (kind_name k.kind));
        ("live_bound", J.i64 (b64 b.live_bound));
        ("initial_upper", J.i64 (b64 b.initial_upper));
        ("lower", J.i64 (b64 b.lower));
        ("upper", J.i64 (b64 b.upper));
        ("gap", J.i64 (Int64.sub (b64 b.upper) (b64 b.lower)));
        ("status", J.str (status_name b.status));
        ("stop", J.str (stop_name b.stop));
        ("queries", J.i64 b.queries);
        ("states", J.i64 r.states);
        ("max_depth", J.i64 r.max_depth);
        ("seconds", J.num r.seconds);
        ("digest", J.str r.digest);
      ])

let strategy_row ids (k : Kind_result.t) (r : Arena_eval.Row.t) =
  let b = k.reference.bounds in
  let p = b64 r.pool in
  let gaps =
    if Int64.equal (b64 b.lower) (b64 b.upper) then
      [
        ("excess", J.i64 (Int64.sub p (b64 b.upper)));
        ( "excess_ratio",
          if Int64.equal (b64 b.upper) 0L then J.null
          else
            J.num
              (Int64.to_float (Int64.sub p (b64 b.upper))
              /. Int64.to_float (b64 b.upper)) );
      ]
    else
      [
        ("excess_min", J.i64 (Int64.max 0L (Int64.sub p (b64 b.upper))));
        ("excess_max", J.i64 (Int64.sub p (b64 b.lower)));
      ]
  in
  J.obj
    (ids
    @ [
        ("type", J.str "strategy");
        ("kind", J.str (kind_name k.kind));
        ("method", J.str (Fmt.str "%a" Arena_eval.Method.pp r.method_));
        ("family", J.str (family r.method_));
        ("strategy", J.str (strategy_name r.method_));
        ("iterations", J.opt J.i64 (Option.map fst r.budget));
        ("seed", J.opt J.i64 (Option.map snd r.budget));
        ("constructive_pool", J.i64 (b64 r.constructive_pool));
        ("pool", J.i64 p);
        ( "iterations_used",
          J.opt J.i64
            (Option.map (fun (e : IA.Effort.t) -> e.iterations) r.effort) );
        ( "stop",
          J.opt J.str
            (Option.map
               (fun (e : IA.Effort.t) ->
                 match e.stop with
                 | IA.Stop.Budget_exhausted -> "budget_exhausted"
                 | Lower_bound -> "lower_bound")
               r.effort) );
        ("live_gap", J.i64 (Int64.sub p k.live_bound));
        ("construct_s", J.num r.timing.construct);
        ("search_s", J.num r.timing.search);
        ("check_s", J.num r.timing.check);
        ("digest", J.str r.digest);
        ("reference_status", J.str (status_name b.status));
      ]
    @ gaps)

(* The best checked pool from any row can close the reference's interval; the
   reference row itself is left as it ran. *)
let comparison ids (k : Kind_result.t) =
  let b = k.reference.bounds in
  let best =
    List.fold_left
      (fun acc (r : Arena_eval.Row.t) ->
        match acc with
        | Some (p, _) when Int64.compare p (b64 r.pool) <= 0 -> acc
        | _ -> Some (b64 r.pool, label r))
      None k.rows
  in
  let best_pool, best_label =
    match best with Some x -> x | None -> (b64 b.upper, "reference")
  in
  if Int64.compare best_pool (b64 b.lower) < 0 then
    Error
      (Fmt.str "%s: pool %Ld below the proven lower bound %Ld" best_label
         best_pool (b64 b.lower))
  else
    let upper = Int64.min (b64 b.upper) best_pool in
    let closed = Int64.equal upper (b64 b.lower) in
    let proof =
      if not closed then None
      else if Int64.equal (b64 b.lower) (b64 b.upper) then Some "reference"
      else Some best_label
    in
    Ok
      (J.obj
         (ids
         @ [
             ("type", J.str "comparison");
             ("kind", J.str (kind_name k.kind));
             ("lower", J.i64 (b64 b.lower));
             ("upper", J.i64 upper);
             ("closed", J.bool closed);
             ("proof", J.opt J.str proof);
             ("best_observed", J.i64 best_pool);
             ("best_observed_by", J.str best_label);
             ("placed", J.i64 k.placed);
           ]))

(* Checked [int64] arithmetic for byte aggregates. *)
let add a b =
  if Int64.compare a (Int64.sub Int64.max_int b) <= 0 then Some (Int64.add a b)
  else None

let sum_bytes f kinds =
  List.fold_left
    (fun acc (k : Kind_result.t) -> Option.bind acc (fun acc -> add acc (f k)))
    (Some 0L) kinds

let best_pool (k : Kind_result.t) =
  List.fold_left
    (fun m (r : Arena_eval.Row.t) -> Int64.min m (b64 r.pool))
    (b64 k.reference.bounds.upper)
    k.rows

(* The production planner's pool: the portfolio at the largest budget, first
   seed. *)
let production_pool (run : Run.t) (k : Kind_result.t) =
  let top = List.fold_left Int64.max 0L run.eval.iterations in
  let seed = List.hd run.eval.seeds in
  List.find_map
    (fun (r : Arena_eval.Row.t) ->
      match (r.method_, r.budget) with
      | Arena_eval.Method.Portfolio, Some (i, s)
        when Int64.equal i top && Int64.equal s seed ->
          Some (b64 r.pool)
      | _ -> None)
    k.rows
  |> Option.value ~default:(b64 k.reference.bounds.upper)

let admitted run (k : Kind_result.t) =
  let bytes = production_pool run k in
  Int64.compare
    (Int64.div bytes
       (Element_bytes.to_int64 (Alloc_script.Kind.element_bytes k.kind)))
    Kernel.Limits.Hard.numel
  < 0
  && Int64.compare bytes Kernel.Limits.default.max_bytes <= 0

let aggregate run ids problem kinds =
  let or_fail what = function
    | Some v -> Ok v
    | None -> Error (Fmt.str "%s bytes overflow int64" what)
  in
  let peak what r =
    match Err.payload r with
    | Ok v -> Ok (b64 v)
    | Error (`Peak_bytes_overflow _) -> or_fail what None
  in
  let ( let* ) = Result.bind in
  let* live = or_fail "live bound" (sum_bytes (fun k -> k.live_bound) kinds) in
  let* lower =
    or_fail "lower bound"
      (sum_bytes (fun k -> b64 k.Kind_result.reference.bounds.lower) kinds)
  in
  let* upper =
    or_fail "upper bound"
      (sum_bytes
         (fun k ->
           Int64.min (b64 k.Kind_result.reference.bounds.upper) (best_pool k))
         kinds)
  in
  let* best = or_fail "best observed" (sum_bytes best_pool kinds) in
  let* production =
    or_fail "production" (sum_bytes (production_pool run) kinds)
  in
  let* placed =
    or_fail "placed" (sum_bytes (fun k -> k.Kind_result.placed) kinds)
  in
  let* combined =
    peak "combined bound" (Arena_problem.combined_bound_bytes problem)
  in
  let* outside =
    peak "out-of-arena" (Arena_problem.out_of_arena_bytes problem)
  in
  Ok
    (J.obj
       (ids
       @ [
           ("type", J.str "aggregate");
           ("kinds", J.int (List.length kinds));
           ( "proven",
             J.bool
               (List.for_all
                  (fun k ->
                    Int64.equal
                      (b64 k.Kind_result.reference.bounds.lower)
                      (Int64.min (b64 k.reference.bounds.upper) (best_pool k)))
                  kinds) );
           ("admitted", J.bool (List.for_all (admitted run) kinds));
           ("per_kind_live_bound_bytes", J.i64 live);
           ("combined_live_bound_bytes", J.i64 combined);
           ("lower_bytes", J.i64 lower);
           ("upper_bytes", J.i64 upper);
           ("best_observed_bytes", J.i64 best);
           ("production_bytes", J.i64 production);
           ("placed_bytes", J.i64 placed);
           ("out_of_arena_bytes", J.i64 outside);
         ]))

(* --- artifacts ------------------------------------------------------------ *)

let write_artifact dir (k : Kind_result.t) =
  let placements =
    (k.reference.digest, k.reference.bounds.incumbent)
    :: List.map (fun (r : Arena_eval.Row.t) -> (r.digest, r.placements)) k.rows
    |> List.sort_uniq (fun (a, _) (b, _) -> String.compare a b)
  in
  let json =
    J.obj
      [
        ("problem_digest", J.str k.digest);
        ("kind", J.str (kind_name k.kind));
        ( "placements",
          J.obj
            (List.map
               (fun (digest, ps) ->
                 ( digest,
                   J.list
                     (fun (id, offset) ->
                       J.list Fun.id
                         [
                           J.int (Tensor_id.to_int id);
                           J.i64 (Byte_offset.to_int64 offset);
                         ])
                     ps ))
               placements) );
      ]
  in
  Out_channel.with_open_bin
    (Filename.concat dir (k.digest ^ ".json"))
    (fun oc -> Out_channel.output_string oc (J.encode json))

(* --- one model ------------------------------------------------------------ *)

let now = Unix.gettimeofday
let failed stage diagnostic = X.Outcome.Failed { stage; diagnostic }

let dialect_row ids outcome =
  let status, stage, reason, diagnostic =
    match outcome with
    | X.Outcome.Failed { stage; diagnostic } ->
        ("failed", Some stage, None, Some diagnostic)
    | Prerequisite { stage; diagnostic } ->
        ("prerequisite", Some stage, None, Some diagnostic)
    | Refused { stage; reason; diagnostic } ->
        ("refused", Some stage, Some reason, Some diagnostic)
  in
  J.obj
    (ids
    @ [
        ("type", J.str "dialect");
        ("status", J.str status);
        ("stage", J.opt J.str (Option.map X.Stage.name stage));
        ("reason", J.opt J.str reason);
        ("diagnostic", J.opt J.str diagnostic);
      ])

let evaluated_row ids (e : X.Extracted.t) kinds eval_s =
  J.obj
    (ids
    @ [
        ("type", J.str "dialect");
        ("status", J.str "evaluated");
        ("nodes", J.int e.nodes);
        ("events", J.int e.events);
        ("script_digest", J.str e.script_digest);
        ("kinds", J.list (fun k -> J.str (kind_name k.Kind_result.kind)) kinds);
        ("convert_s", J.num e.convert_s);
        ("dry_run_s", J.num e.dry_run_s);
        ("eval_s", J.num eval_s);
      ])

(* Every row of one evaluated dialect, or the defect that stops it. *)
let evaluated (run : Run.t) ids (e : X.Extracted.t) =
  let start = now () in
  let rec kinds acc = function
    | [] -> Ok (List.rev acc)
    | kp :: rest -> (
        match Err.payload (evaluate_kind ~now run kp) with
        | Ok k -> kinds (k :: acc) rest
        | Error err -> Error (Fmt.str "%a" pp_eval_error err))
  in
  let ( let* ) = Result.bind in
  let* ks = kinds [] (Arena_problem.kinds e.problem) in
  let eval_s = now () -. start in
  let* per_kind =
    List.fold_left
      (fun acc (k : Kind_result.t) ->
        let* acc = acc in
        let* cmp = comparison ids k in
        Ok
          (acc
          @ [ problem_row ids k; reference_row ids k ]
          @ List.map (strategy_row ids k) k.rows
          @ [ cmp ]))
      (Ok []) ks
  in
  let* agg = aggregate run ids e.problem ks in
  Option.iter (fun dir -> List.iter (write_artifact dir) ks) run.artifacts;
  Ok ((evaluated_row ids e ks eval_s :: per_kind) @ [ agg ])

let model_rows (run : Run.t) (entry : Entry.t) =
  let bytes = In_channel.with_open_bin entry.path In_channel.input_all in
  let ids d =
    [ ("model", J.str entry.model); ("dialect", J.str (X.Dialect.name d)) ]
  in
  let guard f =
    try f ()
    with exn -> Error (failed X.Stage.Internal (Printexc.to_string exn))
  in
  let shared, import_s =
    match
      guard (fun () ->
          Result.map_error (fun (o, _) -> o) (X.canonical ~now bytes))
    with
    | Ok (t, s) -> (Ok t, s)
    | Error o -> (Error o, 0.)
  in
  let model_row =
    J.obj
      [
        ("type", J.str "model");
        ("model", J.str entry.model);
        ("source_md5", J.str entry.md5);
        ("source_bytes", J.int (String.length bytes));
        ( "import_status",
          J.str (match shared with Ok _ -> "ok" | Error _ -> "failed") );
        ("import_s", J.num import_s);
      ]
  in
  let dialect d =
    let ids = ids d in
    match shared with
    | Error o ->
        (* The Native branch owns the import stage; any other only needed it. *)
        [
          dialect_row ids (if d = X.Dialect.Native then o else X.prerequisite o);
        ]
    | Ok transformed -> (
        match guard (fun () -> X.dialect ~now d transformed) with
        | Error o -> [ dialect_row ids o ]
        | Ok e -> (
            match
              try evaluated run ids e
              with exn -> Error ("internal: " ^ Printexc.to_string exn)
            with
            | Ok rows -> rows
            | Error msg -> [ dialect_row ids (failed X.Stage.Evaluate msg) ]))
  in
  (model_row :: List.concat_map dialect run.dialects)
  @ [
      J.obj
        [
          ("type", J.str "model_done");
          ("model", J.str entry.model);
          ("source_md5", J.str entry.md5);
        ];
    ]

(* --- the run -------------------------------------------------------------- *)

let header (run : Run.t) ~source entries =
  let e = run.eval in
  let rev args = Option.map String.trim (command_output "git" args) in
  J.obj
    [
      ("type", J.str "header");
      ("version", J.int version);
      ( "config",
        J.obj
          [
            ("models_dir", J.str run.models_dir);
            ("expected_models", J.int run.expected_models);
            ("dialects", J.list (fun d -> J.str (X.Dialect.name d)) run.dialects);
            ("stage", J.str "normalized");
            ( "normalization",
              J.str "Pipeline.canonical ~fold:false, no payload constants" );
            ( "native4d_conversion",
              J.str "Lower.convert, symbolic constant store only" );
            ("retain", J.str "Only {}");
            ("iterations", J.list J.i64 e.iterations);
            ("seeds", J.list J.i64 e.seeds);
            ("repeats", J.int e.repeats);
            ( "reference",
              J.obj
                [
                  ("max_states", J.i64 e.reference.max_states);
                  ("max_depth", J.i64 e.reference.max_depth);
                ] );
          ] );
      ( "revisions",
        J.obj
          [
            ("repo", J.opt J.str (rev [ "rev-parse"; "HEAD" ]));
            ( "repo_dirty",
              J.opt J.bool
                (Option.map
                   (fun s -> s <> "")
                   (rev
                      [
                        "status";
                        "--porcelain";
                        "--untracked-files=no";
                        "--";
                        "lib";
                        "bin";
                      ])) );
            ( "models",
              J.opt J.str (rev [ "-C"; run.models_dir; "rev-parse"; "HEAD" ]) );
          ] );
      ( "manifest",
        J.obj
          [
            ("source", J.str source);
            ("count", J.int (List.length entries));
            ( "digest",
              J.str
                (Digest.to_hex
                   (Digest.string
                      (String.concat ";"
                         (List.map
                            (fun (e : Entry.t) -> e.model ^ ":" ^ e.md5)
                            entries)))) );
          ] );
      ( "platform",
        J.obj
          [
            ("os_type", J.str Sys.os_type);
            ("ocaml", J.str Sys.ocaml_version);
            ("word_size", J.int Sys.word_size);
          ] );
    ]

(* The completed models of an earlier run with the same header, as a prefix of
   the manifest, and the rest still to do. *)
let resume ~header_line entries output =
  if not (Sys.file_exists output) then Ok ([], entries)
  else
    match lines (In_channel.with_open_bin output In_channel.input_all) with
    | [] -> Ok ([], entries)
    | first :: _ when first <> header_line ->
        Error
          "the existing output was written with a different configuration, \
           revision or corpus"
    | _ :: rest ->
        let rec go kept pending entries = function
          | [] -> Ok (List.rev kept, entries)
          | line :: rest -> (
              let pending = line :: pending in
              let row =
                match J.decode line with Ok j -> Some j | Error _ -> None
              in
              match Option.bind row (J.string_field "type") with
              | Some "model_done" -> (
                  let model = Option.bind row (J.string_field "model")
                  and md5 = Option.bind row (J.string_field "source_md5") in
                  match entries with
                  | (e : Entry.t) :: entries
                    when model = Some e.model && md5 = Some e.md5 ->
                      go (pending @ kept) [] entries rest
                  | _ ->
                      Error
                        (Fmt.str
                           "the existing output's model %s does not match the \
                            manifest"
                           (Option.value model ~default:"?")))
              | _ -> go kept pending entries rest)
        in
        go [] [] entries rest

let write_line oc json =
  Out_channel.output_string oc (J.encode json);
  Out_channel.output_char oc '\n'

let main (run : Run.t) =
  let source, entries = manifest run.models_dir in
  match validate_manifest ~expected:run.expected_models entries with
  | Error msg ->
      Fmt.epr "arena_alloc_eval: %s@." msg;
      1
  | Ok () -> (
      let header_line = J.encode (header run ~source entries) in
      let prior =
        if run.resume then resume ~header_line entries run.output
        else Ok ([], entries)
      in
      match prior with
      | Error msg ->
          Fmt.epr "arena_alloc_eval: cannot resume: %s@." msg;
          1
      | Ok (kept, todo) ->
          Option.iter
            (fun dir -> if not (Sys.file_exists dir) then Sys.mkdir dir 0o755)
            run.artifacts;
          Out_channel.with_open_bin run.output (fun oc ->
              List.iter
                (fun l ->
                  Out_channel.output_string oc l;
                  Out_channel.output_char oc '\n')
                (header_line :: kept);
              Out_channel.flush oc;
              List.iter
                (fun (e : Entry.t) ->
                  Fmt.epr "arena_alloc_eval: %s@." e.model;
                  List.iter (write_line oc) (model_rows run e);
                  Out_channel.flush oc)
                todo);
          let rows =
            List.filter_map
              (fun l ->
                match J.decode l with Ok j -> Some j | Error _ -> None)
              (lines (In_channel.with_open_bin run.output In_channel.input_all))
          in
          let report =
            Arena_eval_summary.of_rows ~expected:run.expected_models
              ~dialects:(List.map X.Dialect.name run.dialects)
              rows
          in
          Out_channel.with_open_gen [ Open_append; Open_binary ]
            0o644 run.output (fun oc ->
              List.iter (write_line oc)
                (Arena_eval_summary.coverage_rows report));
          Option.iter
            (fun path ->
              Out_channel.with_open_bin path (fun oc ->
                  Out_channel.output_string oc
                    (Arena_eval_summary.markdown report)))
            run.summary;
          Option.iter
            (fun path ->
              Out_channel.with_open_bin path (fun oc ->
                  List.iter (write_line oc)
                    (Arena_eval_summary.model_rows report)))
            run.models_output;
          if Arena_eval_summary.ok report then 0 else 1)

(* --- the command line ----------------------------------------------------- *)

open Cmdliner

let int64_list =
  Arg.list
    (Arg.conv'
       ( (fun s ->
           match Int64.of_string_opt s with
           | Some v when Int64.compare v 0L >= 0 -> Ok v
           | _ -> Error (Fmt.str "%S is not a non-negative integer" s)),
         fun ppf v -> Fmt.pf ppf "%Ld" v ))

let dialect =
  Arg.conv'
    ( (fun s ->
        match X.Dialect.of_name s with
        | Some d -> Ok d
        | None -> Error (Fmt.str "unknown dialect %S" s)),
      fun ppf d -> Fmt.string ppf (X.Dialect.name d) )

let term =
  let models_dir =
    Arg.(
      required
      & opt (some dir) None
      & info [ "models-dir" ] ~docv:"DIR"
          ~doc:"The directory of <model>/models/model.json files.")
  and expected =
    Arg.(
      required
      & opt (some int) None
      & info [ "expected-models" ] ~docv:"N"
          ~doc:"The number of models the corpus must hold.")
  and dialects =
    Arg.(
      value
      & opt (list dialect) X.Dialect.all
      & info [ "dialects" ] ~docv:"LIST" ~doc:"native, native4d, or both.")
  and stage =
    Arg.(
      value
      & opt (enum [ ("normalized", ()) ]) ()
      & info [ "stage" ] ~doc:"Only $(b,normalized) is measured.")
  and iterations =
    Arg.(
      value
      & opt int64_list [ 0L; 25L; 100L; 400L ]
      & info [ "iterations" ] ~docv:"LIST" ~doc:"Order-search budgets.")
  and seeds =
    Arg.(value & opt int64_list [ 1L ] & info [ "seeds" ] ~docv:"LIST")
  and repeats =
    Arg.(
      value & opt int 1
      & info [ "repeats" ] ~docv:"N"
          ~doc:"Timing samples per measurement; the fastest is kept.")
  and max_states =
    Arg.(
      required
      & opt (some int64) None
      & info [ "reference-max-states" ] ~docv:"N"
          ~doc:"Expanded-state limit of each kind's reference search.")
  and max_depth =
    Arg.(
      required
      & opt (some int64) None
      & info [ "reference-max-depth" ] ~docv:"N"
          ~doc:"Branch-depth limit of each kind's reference search.")
  and output =
    Arg.(required & opt (some string) None & info [ "output" ] ~docv:"FILE")
  and summary =
    Arg.(value & opt (some string) None & info [ "summary" ] ~docv:"FILE")
  and models_output =
    Arg.(
      value
      & opt (some string) None
      & info [ "models-output" ] ~docv:"FILE"
          ~doc:
            "Write one JSONL row per evaluated model and dialect: the minimum \
             arena and every strategy's arena at every budget, in bytes, \
             summed over element kinds.")
  and artifacts =
    Arg.(
      value
      & opt (some string) None
      & info [ "artifacts" ] ~docv:"DIR"
          ~doc:"Write every checked placement, keyed by problem digest.")
  and resume =
    Arg.(
      value & flag
      & info [ "resume" ]
          ~doc:
            "Keep the completed models of an existing $(b,--output) written \
             with the same configuration, revisions and corpus.")
  in
  let make models_dir expected_models dialects () iterations seeds repeats
      max_states max_depth output summary models_output artifacts resume =
    if seeds = [] || iterations = [] || repeats < 1 then begin
      Fmt.epr
        "arena_alloc_eval: --iterations and --seeds need a value, --repeats >= \
         1@.";
      2
    end
    else
      main
        {
          Run.models_dir;
          expected_models;
          dialects;
          eval =
            {
              Arena_eval.Config.iterations;
              seeds;
              repeats;
              reference = { IA.Reference.Limits.max_states; max_depth };
            };
          output;
          summary;
          models_output;
          artifacts;
          resume;
        }
  in
  Term.(
    const make $ models_dir $ expected $ dialects $ stage $ iterations $ seeds
    $ repeats $ max_states $ max_depth $ output $ summary $ models_output
    $ artifacts $ resume)

let () =
  (match Err_host.install_from_env () with
  | Ok (_ : Err.Config.t option) -> ()
  | Error e ->
      Fmt.epr "arena_alloc_eval: %a@." Err_host.pp_error (Err.Error.kind e);
      exit 124);
  exit
    (Cmd.eval'
       (Cmd.v
          (Cmd.info "arena_alloc_eval"
             ~doc:"Evaluate the arena allocators over the model.json corpus.")
          term))
