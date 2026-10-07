open Machine_ir

module Section = struct
  type t = Bound | Bss | Rodata of string

  let name = function Bound -> "bound" | Bss -> "bss" | Rodata _ -> "rodata"
end

module Symbol = struct
  type kind =
    | Data of {
        region : Mir_id.Region.t;
        size : int64;
        align : int64;
        section : Section.t;
      }
    | External_function of string
    | Function of Mir_id.Func.t

  type t = { name : string; kind : kind }
end

module Relocation = struct
  type t = {
    func : Mir_id.Func.t;
    block : Mir_id.Block.t;
    instr : Mir_id.Instr.t option;
    reference : Mir_target.Reference.t;
    symbol : string;
    addend : int64;
  }
end

module Identity = struct
  type t = {
    target : string;
    source : Mir_target.Source.t;
    features : Mir_target.Feature.t list;
    planning : Mir_planning.t;
    helpers : Mir_helper.t list;
  }
end

type ('op, 'test) t = {
  identity : Identity.t;
  symbols : Symbol.t list;
  relocations : Relocation.t list;
  program : ('op, 'test) Mir_phys.Program.t;
  origins : (Mir_id.Instr.t * Mir_origin.t) list;
}

let identity t = t.identity
let symbols t = t.symbols
let relocations t = t.relocations
let program t = t.program
let origins t = t.origins

let pp_summary fmt t =
  let i = t.identity in
  let count p = List.length (List.filter p t.symbols) in
  let section s =
    count (fun (y : Symbol.t) ->
        match y.Symbol.kind with
        | Symbol.Data { section; _ } -> Section.name section = Section.name s
        | Symbol.External_function _ | Symbol.Function _ -> false)
  in
  Fmt.pf fmt
    "%s (%s %s) features [%s], %a; symbols: %d bound, %d bss, %d rodata, %d \
     external [%s], %d function; %d relocations"
    i.Identity.target i.Identity.source.Mir_target.Source.document
    i.Identity.source.Mir_target.Source.revision
    (String.concat ", " (List.map Mir_target.Feature.name i.Identity.features))
    (fun fmt p ->
      Fmt.pf fmt "%s/%s/%s" p.Mir_planning.policy p.Mir_planning.schedule
        (Mir_planning.Fma.name p.Mir_planning.fma))
    i.Identity.planning (section Section.Bound) (section Section.Bss)
    (section (Section.Rodata ""))
    (count (fun y ->
         match y.Symbol.kind with
         | Symbol.External_function _ -> true
         | _ -> false))
    (String.concat ", "
       (List.filter_map
          (fun (y : Symbol.t) ->
            match y.Symbol.kind with
            | Symbol.External_function s -> Some s
            | _ -> None)
          t.symbols))
    (count (fun y ->
         match y.Symbol.kind with Symbol.Function _ -> true | _ -> false))
    (List.length t.relocations)

let region_symbol r = Fmt.str "mir_region_%d" (Mir_id.Region.to_int r)
let func_symbol (f : (_, _) Mir_phys.Func.t) = f.Mir_phys.Func.name

(* A helper's native binding: only the named math helpers have one. *)
let binding (h : Mir_helper.t) =
  List.find_map
    (fun fn ->
      let d = Mir_math.descriptor fn in
      if
        String.equal d.Mir_helper.name h.Mir_helper.name
        && d.Mir_helper.version = h.Mir_helper.version
      then Some (Mir_math.libm_symbol fn)
      else None)
    Mir_math.Fn.all

module Make (T : Mir_sel.TARGET) = struct
  module S = Mir_sel.Make (T)
  module V = Mir_phys_verify.Make (T)
  module C = Machine_check.Mir_checker.Make (T)

  let ( let* ) = Result.bind

  let publish ~planning (sel : S.Verified.t)
      (p : (T.op, T.test) Mir_phys.Program.t) =
    let* () =
      match
        List.find_opt
          (fun (f : (_, _) Mir_phys.Func.t) ->
            Option.is_none f.Mir_phys.Func.frame)
          p.Mir_phys.Program.funcs
      with
      | Some f ->
          Error
            (Fmt.str "%a has no realized frame" Mir_id.Func.pp
               f.Mir_phys.Func.id)
      | None -> Ok ()
    in
    let* p =
      Result.map_error
        (Fmt.str "physical verifier: %a" Mir_diagnostic.pp)
        (Err.payload (V.verify p))
    in
    let* () =
      Result.map_error
        (Fmt.str "checker: %a" Machine_check.Mir_checker.pp_error)
        (Err.payload (C.check sel p))
    in
    let* helpers =
      List.fold_right
        (fun (h : Mir_helper.t) acc ->
          let* acc = acc in
          match binding h with
          | Some s -> Ok ((h, s) :: acc)
          | None ->
              Error
                (Fmt.str "helper %s (version %d) has no native binding"
                   h.Mir_helper.name h.Mir_helper.version))
        p.Mir_phys.Program.helpers (Ok [])
    in
    let data =
      List.map
        (fun (r : Mir_region.t) ->
          {
            Symbol.name = region_symbol r.Mir_region.id;
            kind =
              Symbol.Data
                {
                  region = r.Mir_region.id;
                  size = r.Mir_region.size;
                  align = r.Mir_region.align;
                  section =
                    (match r.Mir_region.init with
                    | Mir_region.Bound -> Section.Bound
                    | Mir_region.Constant s -> Section.Rodata s
                    | Mir_region.Uninitialized -> Section.Bss);
                };
          })
        p.Mir_phys.Program.regions
    in
    let functions =
      List.map
        (fun (f : (_, _) Mir_phys.Func.t) ->
          {
            Symbol.name = func_symbol f;
            kind = Symbol.Function f.Mir_phys.Func.id;
          })
        p.Mir_phys.Program.funcs
    in
    let externals =
      List.map
        (fun (_, s) -> { Symbol.name = s; kind = Symbol.External_function s })
        helpers
    in
    let view_target v =
      match
        Mir_program.find_view
          {
            Mir_program.data_model = p.Mir_phys.Program.data_model;
            regions = p.Mir_phys.Program.regions;
            views = p.Mir_phys.Program.views;
            helpers = [];
            funcs = [];
            main = p.Mir_phys.Program.main;
            planning = None;
            revision = Mir_id.Revision.of_int 0;
          }
          v
      with
      | Some view ->
          Ok (region_symbol view.Mir_view.region, view.Mir_view.offset)
      | None -> Error (Fmt.str "%a names no view" Mir_id.View.pp v)
    in
    let callee_symbol = function
      | Mir_op.Callee.Func id -> (
          match
            List.find_opt
              (fun (f : (_, _) Mir_phys.Func.t) ->
                Mir_id.Func.equal f.Mir_phys.Func.id id)
              p.Mir_phys.Program.funcs
          with
          | Some f -> Ok (func_symbol f)
          | None ->
              Error (Fmt.str "a call to the undefined %a" Mir_id.Func.pp id))
      | Mir_op.Callee.Helper id -> (
          match
            List.find_opt
              (fun ((h : Mir_helper.t), _) ->
                Mir_id.Helper.equal h.Mir_helper.id id)
              helpers
          with
          | Some (_, s) -> Ok s
          | None ->
              Error (Fmt.str "a call to the undeclared %a" Mir_id.Helper.pp id))
    in
    let* relocations =
      List.fold_right
        (fun (f : (T.op, T.test) Mir_phys.Func.t) acc ->
          List.fold_right
            (fun (b : (T.op, T.test) Mir_phys.Block.t) acc ->
              List.fold_right
                (fun ins acc ->
                  let* acc = acc in
                  let instr, op =
                    match ins with
                    | Mir_phys.Instr.Exec
                        {
                          instr =
                            { Mir_instr.id; op = Mir_sel.Op.Machine op; _ };
                          _;
                        } ->
                        (Some id, Some op)
                    | Mir_phys.Instr.Late { op; _ } -> (None, Some op)
                    | Mir_phys.Instr.Exec _ | Mir_phys.Instr.Move _
                    | Mir_phys.Instr.Save _ | Mir_phys.Instr.Sp _ ->
                        (None, None)
                  in
                  let refs =
                    match op with Some op -> T.references op | None -> []
                  in
                  List.fold_right
                    (fun reference acc ->
                      let* acc = acc in
                      let* symbol, addend =
                        match reference with
                        | Mir_target.Reference.View (v, _) -> view_target v
                        | Mir_target.Reference.Call c ->
                            Result.map (fun s -> (s, 0L)) (callee_symbol c)
                      in
                      Ok
                        ({
                           Relocation.func = f.Mir_phys.Func.id;
                           block = b.Mir_phys.Block.id;
                           instr;
                           reference;
                           symbol;
                           addend;
                         }
                        :: acc))
                    refs (Ok acc))
                b.Mir_phys.Block.body acc)
            f.Mir_phys.Func.blocks acc)
        p.Mir_phys.Program.funcs (Ok [])
    in
    let origins =
      List.concat_map
        (fun (f : (_, _) Mir_phys.Func.t) ->
          List.concat_map
            (fun (b : (_, _) Mir_phys.Block.t) ->
              List.filter_map
                (function
                  | Mir_phys.Instr.Exec { instr; _ } ->
                      Some (instr.Mir_instr.id, instr.Mir_instr.origin)
                  | _ -> None)
                b.Mir_phys.Block.body)
            f.Mir_phys.Func.blocks)
        p.Mir_phys.Program.funcs
    in
    Ok
      {
        identity =
          {
            Identity.target = T.name;
            source = T.source;
            features = p.Mir_phys.Program.features;
            planning;
            helpers = List.map fst helpers;
          };
        symbols = data @ functions @ externals;
        relocations;
        program = p;
        origins;
      }
end
