type t = { slots : (int * int) Expr.Local_var.Map.t; total : int }

let of_locals locals =
  let slots, total =
    List.fold_left
      (fun (slots, offset) local ->
        let count = Region_local.Rhs.slot_count local.Region_local.rhs in
        ( Expr.Local_var.Map.add local.Region_local.id (offset, count) slots,
          offset + count ))
      (Expr.Local_var.Map.empty, 0)
      locals
  in
  { slots; total }

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
