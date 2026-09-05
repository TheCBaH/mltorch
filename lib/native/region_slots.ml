type t = {
  slots : (int * int) Expr.Local_var.Map.t;
  total : int;
  (* [width, steps] for every SCAN-shaped local only -- built once here so
     [scan_reader] below is O(1) setup (a closure over this map), exactly like
     [reader], rather than re-scanning [locals] on every key/output. *)
  scans : (int * int) Expr.Local_var.Map.t;
}

let of_locals locals =
  let slots, total, scans =
    List.fold_left
      (fun (slots, offset, scans) local ->
        let count = Region_local.Rhs.slot_count local.Region_local.rhs in
        let scans =
          match local.Region_local.rhs with
          | Region_local.Rhs.Scan s ->
              Expr.Local_var.Map.add local.Region_local.id
                (s.Expr.Scan.width, s.Expr.Scan.steps)
                scans
          | Region_local.Rhs.Scalar _ | Region_local.Rhs.Vector _ -> scans
        in
        ( Expr.Local_var.Map.add local.Region_local.id (offset, count) slots,
          offset + count,
          scans ))
      (Expr.Local_var.Map.empty, 0, Expr.Local_var.Map.empty)
      locals
  in
  { slots; total; scans }

let total t = t.total
let offset t id = Expr.Local_var.Map.find_opt id t.slots

let reader t values =
  let local id =
    match offset t id with
    | Some (offset, _) when offset < Array.length values -> Some values.(offset)
    | _ -> None
  in
  let local_at id pos =
    match offset t id with
    | Some (offset, count) when pos >= 0 && pos < count ->
        let i = offset + pos in
        if i < Array.length values then Some values.(i) else None
    | _ -> None
  in
  (local, local_at)

(* A cached [Local_scan_at] read: resolve [id] in the SCAN table first (a
   scalar/vector id is not a trace, hence [Unknown_local]), then bounds-check
   row then lane (row wins on a simultaneous failure), then index the same
   flat array [reader] above reads, at the trace's own row-major offset. *)
let scan_reader t values : Expr.Eval.scan_reader =
 fun id ~row ~lane ->
  match Expr.Local_var.Map.find_opt id t.scans with
  | None -> Err.fail (Expr.Eval.Unknown_local id)
  | Some (width, steps) ->
      let offset, _ = Option.get (offset t id) in
      let projection =
        { Expr.Eval.Scan_projection.local = Some id; row; lane }
      in
      if row < 0 || row > steps then
        Err.fail
          (Expr.Eval.Row_out_of_range
             { Expr.Eval.Scan_bounds.projection; extent = steps + 1 })
      else if lane < 0 || lane >= width then
        Err.fail
          (Expr.Eval.Lane_out_of_range
             { Expr.Eval.Scan_bounds.projection; extent = width })
      else Err.return values.(offset + (row * width) + lane)
