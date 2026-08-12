(* Thin wrapper over the [zipc] archive reader, exposing the lookups the pt2
   loader needs. PyTorch writes every entry STORED (uncompressed), but zipc also
   handles Deflate, so this works regardless. Archives are held in memory as a
   string. The single top-level directory all entries nest under is computed
   once at open time (see [prefix]).

   EVERY archive is untrusted, whether a browser or the CLI opened it, so
   [of_string] is bounded by default. The bound is FIVE ORDERED CHECKPOINTS and
   the order is the mitigation, not a style: each one runs before the step that
   would make it too late.

     1. before the OCaml string exists   [Pt2_archive.read_file_limited]
     2. before [Zipc.of_binary_string]   the RAW EOCD member count
     3. immediately after decoding       [Zipc.member_count] = the raw count
     4. over the central directory       paths, encryption, sizes, aggregate
     5. at materialization, per method   [read]

   See .ai/model_explorer_design.md. *)

(* Paths come off the wire and may be up to [Zipc.Member.max_path_length]
   (65535) bytes, so no error payload carries one whole: a diagnostic about an
   oversized input must not itself be one. *)
let path_excerpt_bytes = 64

let excerpt path =
  if String.length path <= path_excerpt_bytes then path
  else String.sub path 0 path_excerpt_bytes ^ "..."

module Limits = struct
  type t = {
    max_entries : int;
    max_entry_bytes : int64;
    max_total_bytes : int64;
    max_path_bytes : int;
    max_path_depth : int;
    allow_encrypted : bool;
  }

  module Invalid = struct
    type t = { name : string; value : int64 }
  end

  type error = [ `Invalid_limit of Invalid.t ]

  let pp_error fmt : [< error ] -> unit = function
    | `Invalid_limit { Invalid.name; value } ->
        Fmt.pf fmt "invalid zip limit %s = %Ld" name value

  module Hard = struct
    (* Fixed from the ZIP format and the platform, never from measuring
       well-behaved models. zipc rejects ZIP64 outright (zipc.ml:290), so the
       classic 16-bit fields ARE the format's ceilings. *)

    (* [Zipc.Member.max]: the EOCD total-entries field is a uint16, so no
       conforming archive can declare more, and our own preflight reads that
       same field. One past it, so [create] can name the format maximum. *)
    let entries = 0x10000

    (* [Zipc.Member.max_path_length]. *)
    let path_bytes = 0x10000

    (* Nothing structural bounds nesting depth; this is a traversal-cost bound,
       and 64 is far past anything PyTorch emits (".../data/weights/<name>"). *)
    let path_depth = 64

    (* [Zipc.File.max_size] is [min Int.max_int 4294967295], and [Int.max_int]
       is where the backends diverge: 2^62-1 natively, 2^31-1 under
       js_of_ocaml (measured under node: [Sys.int_size = 32],
       [max_int = 2147483647], [Sys.max_string_length = 2147483643]). So
       [max_size] is 4294967295 natively and 2147483647 under jsoo — the same
       archive would open in one shell and fail in the other. The ceiling is
       fixed for both at 2^30 bytes, which is half the jsoo string domain, so
       an entry and a copy of it still fit inside it. *)
    let entry_bytes = 0x4000_0000L

    (* The whole archive, decompressed. A .pt2 for a large vision model is tens
       of megabytes; 4GB of declared total is a bomb, not a model, and this
       still leaves three orders of magnitude of headroom. *)
    let total_bytes = 0x1_0000_0000L
  end

  let check_int name v hard =
    if v <= 0 || v > hard then
      Err.fail (`Invalid_limit { Invalid.name; value = Int64.of_int v })
    else Err.return ()

  let check_int64 name v hard =
    if Int64.compare v 0L <= 0 || Int64.compare v hard > 0 then
      Err.fail (`Invalid_limit { Invalid.name; value = v })
    else Err.return ()

  let create ~max_entries ~max_entry_bytes ~max_total_bytes ~max_path_bytes
      ~max_path_depth =
    let open Err.Syntax in
    let* () = check_int "max_entries" max_entries Hard.entries in
    let* () = check_int64 "max_entry_bytes" max_entry_bytes Hard.entry_bytes in
    let* () = check_int64 "max_total_bytes" max_total_bytes Hard.total_bytes in
    let* () = check_int "max_path_bytes" max_path_bytes Hard.path_bytes in
    let+ () = check_int "max_path_depth" max_path_depth Hard.path_depth in
    {
      max_entries;
      max_entry_bytes;
      max_total_bytes;
      max_path_bytes;
      max_path_depth;
      (* Not a parameter: zipc cannot decrypt at all ([Zipc.File.can_extract]),
         so a permissive profile would only defer the failure to a worse place.
         The field exists so the rejection has a name in the record rather than
         being an unexplained constant in the walk. *)
      allow_encrypted = false;
    }

  (* The default for EVERY file-shaped input, CLI included. Provisional and
     conservative: the release profile is calibrated after Stage 2. A real
     resnet18 .pt2 has ~250 entries and ~45MB of payload. *)
  let untrusted =
    Err.or_raise ~pp_error
      (create ~max_entries:4096 ~max_entry_bytes:0x1000_0000L
         ~max_total_bytes:0x8000_0000L ~max_path_bytes:1024 ~max_path_depth:16)

  (* Internal/programmatic callers holding data they produced. Never reachable
     from a file the user chose — there is no [--limits trusted]. *)
  let trusted =
    Err.or_raise ~pp_error
      (create ~max_entries:Hard.entries ~max_entry_bytes:Hard.entry_bytes
         ~max_total_bytes:Hard.total_bytes ~max_path_bytes:Hard.path_bytes
         ~max_path_depth:Hard.path_depth)
end

module Entry_bound = struct
  type kind = Compressed | Decompressed

  type t = { path : string; kind : kind; limit : int64 }
  (** [limit], not the measure, matching [Kernel.error]'s rule: reporting the
      actual figure means trusting a number the archive declared. [path] is an
      [excerpt]. *)

  let kind_name = function
    | Compressed -> "compressed"
    | Decompressed -> "decompressed"
end

module Path_rejection = struct
  type kind =
    | Too_long of int  (** the limit *)
    | Too_deep of int  (** the limit *)
    | Absolute
    | Parent_segment
    | Control_byte

  type t = { path : string; kind : kind }
  (** [path] is an [excerpt]. *)
end

(* Two different facts under one tag before: zipc's own inflate message, and a
   produced-length disagreement this module detects. The second is the whole
   point of the post-check (see [read]), and a caller could only tell them
   apart by matching on the message text. *)
type zip_read_cause = [ `Zipc of string | `Size_disagrees ]

module Zip_read_failure = struct
  type t = { name : string; cause : zip_read_cause }
  (** [name] is an [excerpt], as every other path in this row is. *)
end

type t = { zip : Zipc.t; prefix : string }

type error =
  [ `Zip_parse_failed of string
  | `Zip_read_failed of Zip_read_failure.t
  | `Zip_missing_entry of string
  | `Zip_no_eocd
  | `Zip_too_many_entries of int
  | `Zip_duplicate_paths of int
  | `Zip_bad_path of Path_rejection.t
  | `Zip_encrypted_entry of string
  | `Zip_unsupported_method of string
  | `Zip_entry_too_large of Entry_bound.t
  | `Zip_total_too_large of int64
  | `Zip_stored_size_mismatch of string ]

let entries_of zip = Zipc.fold (fun m acc -> Zipc.Member.path m :: acc) zip []

(* PyTorch nests everything under a single top-level directory ("<model>/" in a
   .pt2, "archive/" in a .pt). Find it from the first member that has one (a bare
   directory member, if any, has no '/'). *)
let compute_prefix zip =
  let rec first = function
    | [] -> ""
    | name :: rest -> (
        match String.index_opt name '/' with
        | Some i -> String.sub name 0 i
        | None -> first rest)
  in
  first (entries_of zip)

let pp_path_rejection ppf ({ path; kind } : Path_rejection.t) =
  match kind with
  | Path_rejection.Too_long limit ->
      Fmt.pf ppf "zip path %S exceeds %d bytes" path limit
  | Path_rejection.Too_deep limit ->
      Fmt.pf ppf "zip path %S is deeper than %d components" path limit
  | Path_rejection.Absolute -> Fmt.pf ppf "zip path %S is absolute" path
  | Path_rejection.Parent_segment ->
      Fmt.pf ppf "zip path %S has a %S component" path Filename.parent_dir_name
  | Path_rejection.Control_byte ->
      Fmt.pf ppf "zip path %S has a control byte" path

let pp_error ppf : error -> unit = function
  | `Zip_parse_failed msg -> Fmt.pf ppf "zip parse failed: %s" msg
  | `Zip_read_failed { Zip_read_failure.name; cause } ->
      Fmt.pf ppf "zip entry %S read failed: %s" name
        (match cause with
        | `Zipc m -> m
        | `Size_disagrees -> "decompressed size disagrees")
  | `Zip_missing_entry name -> Fmt.pf ppf "zip entry %S is missing" name
  | `Zip_no_eocd -> Fmt.pf ppf "zip has no end-of-central-directory record"
  | `Zip_too_many_entries limit ->
      Fmt.pf ppf "zip declares more than %d entries" limit
  | `Zip_duplicate_paths declared ->
      Fmt.pf ppf "zip declares %d entries but has fewer distinct paths" declared
  | `Zip_bad_path r -> pp_path_rejection ppf r
  | `Zip_encrypted_entry path -> Fmt.pf ppf "zip entry %S is encrypted" path
  | `Zip_unsupported_method path ->
      Fmt.pf ppf "zip entry %S uses an unsupported compression method" path
  | `Zip_entry_too_large { Entry_bound.path; kind; limit } ->
      Fmt.pf ppf "zip entry %S declares %s size over %Ld bytes" path
        (Entry_bound.kind_name kind)
        limit
  | `Zip_total_too_large limit ->
      Fmt.pf ppf "zip entries total over %Ld bytes" limit
  | `Zip_stored_size_mismatch path ->
      Fmt.pf ppf "zip entry %S is stored with disagreeing sizes" path

(* CHECKPOINT 2 — before [Zipc.of_binary_string].

   The count zipc will act on, read from the bytes rather than from the decoded
   archive. It has to be this one: [Zipc.fold] and [Zipc.member_count] run over
   a path-keyed map in which two members sharing a path have already collapsed
   into one, so counting there cannot see the entries it is meant to bound.
   Checkpoint 3 turns that same gap into the duplicate detector.

   The search mirrors zipc's own (zipc.ml:410-425) exactly — backwards from
   [len - 22] for at most 65535 comment bytes, first match wins — because a
   preflight that located a DIFFERENT record would bound a different archive. *)
let eocd_sig = 0x06054b50l
let eocd_min_size = 22
let eocd_max_comment = 65535

let raw_eocd_entry_count s =
  let len = String.length s in
  let start = len - eocd_min_size in
  let min_start = len - eocd_max_comment - eocd_min_size in
  let rec loop i =
    if i < min_start || i < 0 then None
    else if Int32.equal (String.get_int32_le s i) eocd_sig then
      Some (String.get_uint16_le s (i + 10))
    else loop (i - 1)
  in
  if start < 0 then None else loop start

(* CHECKPOINT 4 — over the central directory, before anything is materialized.

   Path shape first, then encryption, then the declared sizes, then the running
   total. Nothing here decompresses. *)
let check_path (limits : Limits.t) path =
  let reject kind =
    Err.fail (`Zip_bad_path { Path_rejection.path = excerpt path; kind })
  in
  if String.length path > limits.Limits.max_path_bytes then
    reject (Path_rejection.Too_long limits.Limits.max_path_bytes)
  else if String.length path > 0 && path.[0] = '/' then
    reject Path_rejection.Absolute
  else
    let segments = String.split_on_char '/' path in
    if List.length segments > limits.Limits.max_path_depth then
      reject (Path_rejection.Too_deep limits.Limits.max_path_depth)
    else if List.exists (fun c -> c = Filename.parent_dir_name) segments then
      reject Path_rejection.Parent_segment
    else if
      String.exists (fun c -> Char.code c < 0x20 || Char.code c = 0x7F) path
    then reject Path_rejection.Control_byte
    else Err.return ()

let check_file (limits : Limits.t) path (f : Zipc.File.t) =
  let open Err.Syntax in
  let too_large kind =
    Err.fail
      (`Zip_entry_too_large
         {
           Entry_bound.path = excerpt path;
           kind;
           limit = limits.Limits.max_entry_bytes;
         })
  in
  let* () =
    if not (Zipc.File.is_encrypted f) then Err.return ()
    else if limits.Limits.allow_encrypted then Err.return ()
    else Err.fail (`Zip_encrypted_entry (excerpt path))
  in
  let* () =
    if Zipc.File.can_extract f then Err.return ()
    else Err.fail (`Zip_unsupported_method (excerpt path))
  in
  (* [decompressed_size] and [compressed_size] are [int], which is 32-bit under
     js_of_ocaml, so BOTH widen to [int64] before meeting an [int64] limit — a
     comparison against a value that may already have wrapped is not a bound.
     They cannot actually wrap here (zipc caps them at [File.max_size]), and the
     widening is still first, because that argument is about today's zipc. *)
  let compressed = Int64.of_int (Zipc.File.compressed_size f) in
  let decompressed = Int64.of_int (Zipc.File.decompressed_size f) in
  let* () =
    if Int64.compare compressed limits.Limits.max_entry_bytes > 0 then
      too_large Entry_bound.Compressed
    else Err.return ()
  in
  let+ () =
    if Int64.compare decompressed limits.Limits.max_entry_bytes > 0 then
      too_large Entry_bound.Decompressed
    else Err.return ()
  in
  (* The aggregate counts whichever is LARGER. A Stored member may declare a
     [decompressed_size] smaller than the bytes it actually yields (checkpoint 5
     rejects that pair, but this runs first and must not be fooled meanwhile),
     and a Deflate member's decompressed size is the memory it costs. *)
  if Int64.compare compressed decompressed > 0 then compressed else decompressed

let check_central_directory limits zip =
  let open Err.Syntax in
  let members = Zipc.fold (fun m acc -> m :: acc) zip [] in
  let rec walk total = function
    | [] -> Err.return ()
    | m :: rest ->
        let path = Zipc.Member.path m in
        let* () = check_path limits path in
        let* size =
          match Zipc.Member.kind m with
          | Zipc.Member.Dir -> Err.return 0L
          | Zipc.Member.File f -> check_file limits path f
        in
        (* CHECKED BEFORE THE ADDITION, not inspected after it: a wrapped total
           passes a [<=] test, which is the defect CLAUDE.md records six times.
           Both operands are non-negative and [max_total_bytes] is at most
           [Hard.total_bytes] (2^32), so the subtraction cannot underflow. *)
        let remaining = Int64.sub limits.Limits.max_total_bytes total in
        if Int64.compare size remaining > 0 then
          Err.fail (`Zip_total_too_large limits.Limits.max_total_bytes)
        else walk (Int64.add total size) rest
  in
  walk 0L members

let of_string ?(limits = Limits.untrusted) s =
  let open Err.Syntax in
  (* 2 *)
  let* declared = raw_eocd_entry_count s |> Err.of_option `Zip_no_eocd in
  let* () =
    if declared > limits.Limits.max_entries then
      Err.fail (`Zip_too_many_entries limits.Limits.max_entries)
    else Err.return ()
  in
  let* zip =
    match Zipc.of_binary_string s with
    | Ok zip -> Err.return zip
    | Error e -> Err.fail (`Zip_parse_failed e)
  in
  (* 3 — the ONLY place a duplicate path below [max_entries] is detectable.
     zipc keys members by path, so two entries claiming one path decode to a
     single member and the loser is silently gone; comparing the decoded count
     with the declared one is what turns that back into a rejection. *)
  let* () =
    if Zipc.member_count zip <> declared then
      Err.fail (`Zip_duplicate_paths declared)
    else Err.return ()
  in
  (* 4 *)
  let+ () = check_central_directory limits zip in
  { zip; prefix = compute_prefix zip }

let entries t = entries_of t.zip
let prefix t = t.prefix

(* CHECKPOINT 5 — at materialization, per method.

   Checkpoint 4 has already bounded the DECLARED sizes, so what is left is the
   fact a declaration cannot carry: how many bytes the member really yields.

   For Stored that is [compressed_size], so the pre-check and the post-check
   below test the same equality — deliberately, and they are not redundant. The
   pre-check refuses BEFORE [to_binary_string] builds the string; the post-check
   would refuse after allocating it, which for a member near the per-entry
   ceiling is the whole cost. The suite distinguishes them by their messages,
   which is the only external evidence of which one fired -- or was, before
   [`Size_disagrees] made the post-check its own constructor.

   [to_binary_string] is itself safe on the archive range — decoding already
   validated that each compressed span fits (zipc.ml:335) — so neither check is
   standing between us and an [Invalid_argument]. What they stand between is a
   member and an aggregate that was told a different number. *)
let read t name =
  let open Err.Syntax in
  match Zipc.find name t.zip with
  | None -> Err.return None
  | Some m -> (
      match Zipc.Member.kind m with
      | Zipc.Member.Dir -> Err.return None
      | Zipc.Member.File f ->
          let declared = Zipc.File.decompressed_size f in
          let* () =
            match Zipc.File.compression f with
            | Zipc.Stored ->
                if Zipc.File.compressed_size f <> declared then
                  Err.fail (`Zip_stored_size_mismatch (excerpt name))
                else Err.return ()
            | Zipc.Deflate -> Err.return ()
            | _ -> Err.fail (`Zip_unsupported_method (excerpt name))
          in
          let* s =
            match Zipc.File.to_binary_string f with
            | Ok s -> Err.return s
            | Error e ->
                (* [excerpt], like every sibling check above: the bound on
                   error-payload size was skipped at this site and the next
                   one only. *)
                Err.fail
                  (`Zip_read_failed
                     { Zip_read_failure.name = excerpt name; cause = `Zipc e })
          in
          (* The produced-length re-check. zipc's inflate is bounded by the
             declared size, so this closes the other direction: a member that
             yielded FEWER bytes than the central directory claimed was counted
             for more, which is safe, and one that yielded more would not be. *)
          if String.length s <> declared then
            Err.fail
              (`Zip_read_failed
                 {
                   Zip_read_failure.name = excerpt name;
                   cause = `Size_disagrees;
                 })
          else Err.return (Some s))

let read_required t name =
  let open Err.Syntax in
  let* data = read t name in
  data |> Err.of_option (`Zip_missing_entry name)

let read_rel t rel = read t (t.prefix ^ "/" ^ rel)
let read_rel_required t rel = read_required t (t.prefix ^ "/" ^ rel)
