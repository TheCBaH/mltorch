(* See arena_run.mli. *)

open Graph_ir
open Core.Storage_units

type error =
  [ Arena.error
  | Arena_plan.error
  | Eval_direct.error
  | `Peak_bytes_overflow of Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | #Arena.error as e -> Arena.pp_error ppf e
  | #Eval_direct.error as e -> Eval_direct.pp_error ppf e
  | `Arena_over_limit { Arena_plan.Over_limit.kind; numel; bytes; limit } ->
      Format.fprintf ppf "arena: the %a pool needs %a cells (%a bytes), over %a"
        Alloc_script.Kind.pp kind Element_count.pp numel Byte_size.pp bytes
        (fun ppf -> function
          | Arena_plan.Over_limit.Bytes b ->
              Format.fprintf ppf "%a bytes" Byte_size.pp b
          | Arena_plan.Over_limit.Elements n ->
              Format.fprintf ppf "%a cells" Element_count.pp n)
        limit
  | `Arena_placement _ -> Format.pp_print_string ppf "arena: placement failed"
  | `Arena_script id ->
      Format.fprintf ppf "arena: inconsistent script at t%d"
        (Tensor_id.to_int id)
  | `Peak_bytes_overflow id ->
      Format.fprintf ppf "arena: byte total overflows at t%d"
        (Tensor_id.to_int id)

type outcome = Arena of Arena.t | Release_only of error

let plan_and_create ?limits ?budget ?alignment
    ?(physical = Arena.Physical_requirement.Logical_accepted) ?poison ?retain
    ~admission g =
  let open Err.Syntax in
  let* script =
    Eval_direct.dry_run ?alignment ?retain g
    |> Err.map_error (fun e -> (e :> error))
  in
  let* plan =
    Arena_plan.create ?limits ?budget ?alignment script
    |> Err.map_error (fun e -> (e :> error))
  in
  let* () =
    match (physical, Arena_plan.base_alignment plan) with
    | Arena.Physical_requirement.Physical_required, Some required -> (
        match Arena.Physical_alignment.backend with
        | Arena.Physical_alignment.Logical_only ->
            Err.fail ~pos:__POS__
              (`Physical_alignment_unsupported
                 {
                   Arena.Physical_unsupported.required;
                   provided = Arena.Physical_alignment.backend;
                 }))
    | Arena.Physical_requirement.Physical_required, None
    | Arena.Physical_requirement.Logical_accepted, _ ->
        Err.return ()
  in
  let* () =
    match admission with
    | Arena.Admission.Best_effort -> Err.return ()
    | Arena.Admission.Required budget ->
        let* footprint =
          Arena.footprint plan |> Err.map_error (fun e -> (e :> error))
        in
        if Byte_size.compare footprint budget > 0 then
          Err.fail ~pos:__POS__
            (`Over_budget { Arena.Over_budget.footprint; budget })
        else Err.return ()
  in
  Arena.create ?poison plan |> Err.map_error (fun e -> (e :> error))

(* The row is data for the caller's report, not an error of this run: under
   [Best_effort] a failure here means the run goes ahead without an arena. *)
let decline (e : error Err.Error.t) =
  match Err.export ~pos:__POS__ (Error e) with
  | Error row -> Release_only row
  | Ok _ -> assert false

let acquire ?limits ?budget ?alignment ?physical ?poison ?retain ~admission g =
  match
    ( plan_and_create ?limits ?budget ?alignment ?physical ?poison ?retain
        ~admission g,
      admission )
  with
  | Ok arena, _ -> Err.return (Arena arena)
  | Error e, Arena.Admission.Best_effort -> Err.return (decline e)
  | (Error _ as e), Arena.Admission.Required _ -> (e :> (outcome, error) Err.t)

module Report = struct
  type t = {
    pool_bytes : Byte_size.t;
    out_of_arena_bytes : Byte_size.t;
    base_alignment : Byte_alignment.t option;
    physical_alignment : Arena.Physical_alignment.t;
    copies : Arena.Copies.t;
  }

  let pp ppf t =
    Format.fprintf ppf
      "pool_bytes=%a out_of_arena_bytes=%a base_alignment=%a \
       physical_alignment=%a mixed_mode_copies=%Ld (%a bytes)"
      Byte_size.pp t.pool_bytes Byte_size.pp t.out_of_arena_bytes
      (Fmt.option ~none:(Fmt.any "none") Byte_alignment.pp)
      t.base_alignment Arena.Physical_alignment.pp t.physical_alignment
      t.copies.Arena.Copies.count Byte_size.pp t.copies.Arena.Copies.bytes
end

let report arena =
  let open Err.Syntax in
  let plan = Arena.plan arena in
  let+ out_of_arena_bytes =
    Alloc_script.out_of_arena_bytes (Arena_plan.script plan)
  in
  {
    Report.pool_bytes = (Arena_plan.stats plan).Arena_plan.Stats.pool_bytes;
    out_of_arena_bytes;
    base_alignment = Arena_plan.base_alignment plan;
    physical_alignment = Arena.physical_alignment arena;
    copies = Arena.copies arena;
  }

module Outcome = struct
  type t = Declined of error | Used of Report.t
end

let with_arena ?limits ?budget ?alignment ?physical ?poison ?retain ~admission g
    (f : Arena.t option -> ('a, error) Err.t) : ('a * Outcome.t, error) Err.t =
  let open Err.Syntax in
  let* outcome =
    acquire ?limits ?budget ?alignment ?physical ?poison ?retain ~admission g
  in
  match outcome with
  | Release_only reason ->
      let+ r = f None in
      (r, Outcome.Declined reason)
  | Arena arena ->
      let* r = f (Some arena) in
      let+ used =
        Err.map_error (fun (`Peak_bytes_overflow _ as e) -> e) (report arena)
      in
      (r, Outcome.Used used)
