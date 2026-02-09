from __future__ import annotations


def gds_import_script(
    *,
    gds_path: str,
    cell_name: str,
    layer_map: str = "1:0",
) -> str:
    # This is a minimal template; for real runs align with `data_CR/data_gen.ipynb`.
    # LumAPI scripts are executed via fdtd.eval(script).
    gds_path_escaped = gds_path.replace("\\", "/")
    return (
        f"gds_path = '{gds_path_escaped}';\n"
        f"cell_name = '{cell_name}';\n"
        f"layer_map = '{layer_map}';\n"
        "switchtolayout;\n"
        "gdsimport(gds_path, cell_name, layer_map);\n"
    )


def extract_spectra_script() -> str:
    # Placeholder extract. Real script depends on template monitor/result names.
    # If you know the correct monitor/object names in `air_SiN_2um_NA.fsp`, update here.
    return (
        "# TODO: replace with template-specific extraction.\n"
        "# Example expects variables T1,T2,T3,f_vec to be assigned.\n"
        "f_vec = getdata('monitor','f');\n"
        "T1 = getdata('monitor','T1');\n"
        "T2 = getdata('monitor','T2');\n"
        "T3 = getdata('monitor','T3');\n"
    )
