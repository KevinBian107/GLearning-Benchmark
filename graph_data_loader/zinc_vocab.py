"""
Shared ZINC vocabulary for consistent feature encoding across IBTT and AGTT.

This module defines a FIXED vocabulary mapping for ZINC molecular features
to ensure both Index-Based Tokenization (IBTT) and AutoGraph Trail Tokenization (AGTT)
use the SAME embedding indices for the same chemical features.

Key insight: By using a fixed mapping, Carbon→ID X, Nitrogen→ID Y, etc.,
both methods will look up embeddings from the same indices, enabling fair comparison.
"""

from typing import Dict, Tuple


# =============================================================================
# ZINC Feature Constants
# =============================================================================

# ZINC atom types (9 types, matching PyG ZINC dataset)
ZINC_ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
NUM_ATOM_TYPES = len(ZINC_ATOM_TYPES)  # 9

# ZINC bond types (4 types)
ZINC_BOND_TYPES = ['single', 'double', 'triple', 'aromatic']
NUM_BOND_TYPES = len(ZINC_BOND_TYPES)  # 4

# Special tokens (matching data_loader.py SPECIAL)
SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', '<unk>', '<q>', '<p>', '<atom>', '<bond>']


# =============================================================================
# Fixed Vocabulary Construction
# =============================================================================

def build_fixed_zinc_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a fixed vocabulary for ZINC with deterministic token IDs.

    Token ID structure:
    - 0-7: Special tokens (<bos>, <eos>, <pad>, <unk>, <q>, <p>, <atom>, <bond>)
    - 8-16: Atom types (C=8, N=9, O=10, F=11, P=12, S=13, Cl=14, Br=15, I=16)
    - 17-20: Bond types (single=17, double=18, triple=19, aromatic=20)
    - 21: Task token for 'regression'
    - 22+: Reserved for node indices, label values, etc. (dynamically added during training)

    This fixed structure ensures:
    - IBTT: "C" → ID 8 → embed.weight[8]
    - AGTT: atom_type_0 (Carbon) → ID 8 → embed.weight[8]

    Returns:
        (vocab, itos): token→ID mapping and ID→token reverse mapping
    """
    vocab = {}
    idx = 0

    # 1. Special tokens (IDs 0-7)
    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1

    # 2. Atom types (IDs 8-16)
    for atom in ZINC_ATOM_TYPES:
        vocab[atom] = idx
        idx += 1

    # 3. Bond types (IDs 17-20)
    for bond in ZINC_BOND_TYPES:
        vocab[bond] = idx
        idx += 1

    # 4. Task token (ID 21)
    vocab['regression'] = idx
    idx += 1

    # Reverse mapping
    itos = {i: t for t, i in vocab.items()}

    return vocab, itos


def get_atom_type_id(atom_type_idx: int) -> int:
    """
    Get fixed token ID for an atom type index.

    Args:
        atom_type_idx: Atom type index from PyG ZINC (0-8)
                      0=C, 1=N, 2=O, 3=F, 4=P, 5=S, 6=Cl, 7=Br, 8=I

    Returns:
        Fixed token ID (8-16)
    """
    if not (0 <= atom_type_idx < NUM_ATOM_TYPES):
        raise ValueError(f"Invalid atom type index: {atom_type_idx} (expected 0-{NUM_ATOM_TYPES-1})")

    # Atom types start at ID 8 (after special tokens)
    return 8 + atom_type_idx


def get_bond_type_id(bond_type_idx: int) -> int:
    """
    Get fixed token ID for a bond type index.

    Args:
        bond_type_idx: Bond type index from PyG ZINC (1-4)
                      1=single, 2=double, 3=triple, 4=aromatic

    Returns:
        Fixed token ID (17-20)
    """
    if not (1 <= bond_type_idx <= NUM_BOND_TYPES):
        raise ValueError(f"Invalid bond type index: {bond_type_idx} (expected 1-{NUM_BOND_TYPES})")

    # Bond types start at ID 17 (after special tokens + atom types)
    # Bond indices are 1-based, so subtract 1
    return 17 + (bond_type_idx - 1)


def get_atom_type_from_id(token_id: int) -> str:
    """
    Get atom type string from token ID.

    Args:
        token_id: Token ID (8-16)

    Returns:
        Atom type string ('C', 'N', etc.)
    """
    if not (8 <= token_id <= 16):
        raise ValueError(f"Invalid atom type token ID: {token_id} (expected 8-16)")

    return ZINC_ATOM_TYPES[token_id - 8]


def get_bond_type_from_id(token_id: int) -> str:
    """
    Get bond type string from token ID.

    Args:
        token_id: Token ID (17-20)

    Returns:
        Bond type string ('single', 'double', etc.)
    """
    if not (17 <= token_id <= 20):
        raise ValueError(f"Invalid bond type token ID: {token_id} (expected 17-20)")

    return ZINC_BOND_TYPES[token_id - 17]


# =============================================================================
# Vocabulary Extension for Dynamic Tokens
# =============================================================================

def extend_vocab_with_dynamic_tokens(
    base_vocab: Dict[str, int],
    dynamic_tokens: list
) -> Dict[str, int]:
    """
    Extend the fixed vocabulary with dynamic tokens (node indices, label values, etc.).

    This is needed for IBTT which tokenizes node indices as strings ("0", "1", ...)
    and regression labels ("val_4_23", etc.).

    Args:
        base_vocab: Fixed vocabulary from build_fixed_zinc_vocab()
        dynamic_tokens: List of additional tokens to add

    Returns:
        Extended vocabulary
    """
    vocab = base_vocab.copy()
    idx = max(vocab.values()) + 1

    for token in dynamic_tokens:
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    return vocab


# =============================================================================
# Mapping for AGTT (AutoGraph Tokenization)
# =============================================================================

def map_autograph_token_to_fixed_id(
    autograph_token_id: int,
    tokenizer_node_idx_offset: int,
    tokenizer_edge_idx_offset: int,
    is_node_type: bool = False,
    is_edge_type: bool = False,
) -> int:
    """
    Map AutoGraph's tokenizer output to fixed vocabulary IDs.

    AutoGraph uses structured offsets:
    - Special tokens: 0-5 (SOS, RESET, LADJ, RADJ, EOS, PAD)
    - Node positions: tokenizer.idx_offset to idx_offset + max_nodes - 1
    - Atom types: tokenizer.node_idx_offset to node_idx_offset + num_atom_types - 1
    - Bond types: tokenizer.edge_idx_offset to edge_idx_offset + num_bond_types - 1

    This function remaps AutoGraph's atom/bond type IDs to our fixed vocabulary.

    Args:
        autograph_token_id: Token ID from AutoGraph tokenizer
        tokenizer_node_idx_offset: AutoGraph's node_idx_offset
        tokenizer_edge_idx_offset: AutoGraph's edge_idx_offset
        is_node_type: If True, interpret as atom type
        is_edge_type: If True, interpret as bond type

    Returns:
        Fixed vocabulary token ID
    """
    # Special tokens mapping (AutoGraph → Fixed vocab)
    # AutoGraph: SOS=0, RESET=1, LADJ=2, RADJ=3, EOS=4, PAD=5
    # Fixed: <bos>=0, <eos>=1, <pad>=2, <unk>=3, <q>=4, <p>=5
    autograph_special_to_fixed = {
        0: 0,  # SOS → <bos>
        4: 1,  # EOS → <eos>
        5: 2,  # PAD → <pad>
    }

    if autograph_token_id in autograph_special_to_fixed:
        return autograph_special_to_fixed[autograph_token_id]

    # Atom types (node types)
    if is_node_type:
        atom_type_idx = autograph_token_id - tokenizer_node_idx_offset
        return get_atom_type_id(atom_type_idx)

    # Bond types (edge types)
    if is_edge_type:
        bond_type_idx = autograph_token_id - tokenizer_edge_idx_offset + 1  # +1 because bond indices are 1-based
        return get_bond_type_id(bond_type_idx)

    # For node positions and other tokens, keep as-is for now
    # (they don't affect chemistry encoding comparison)
    return autograph_token_id


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    # Build fixed vocabulary
    vocab, itos = build_fixed_zinc_vocab()

    print("Fixed ZINC Vocabulary")
    print("=" * 80)
    print("\nSpecial Tokens:")
    for i in range(8):
        print(f"  {i}: {itos[i]}")

    print("\nAtom Types:")
    for i in range(8, 17):
        print(f"  {i}: {itos[i]}")

    print("\nBond Types:")
    for i in range(17, 21):
        print(f"  {i}: {itos[i]}")

    print(f"\nTask Token:")
    print(f"  {vocab['regression']}: regression")

    print("\n" + "=" * 80)
    print("\nVerification:")
    print(f"  Carbon (atom type 0) → ID {get_atom_type_id(0)}")
    print(f"  Nitrogen (atom type 1) → ID {get_atom_type_id(1)}")
    print(f"  Single bond (bond type 1) → ID {get_bond_type_id(1)}")
    print(f"  Double bond (bond type 2) → ID {get_bond_type_id(2)}")

    print("\n" + "=" * 80)
    print("\nThis ensures both IBTT and AGTT use the SAME embedding indices!")
    print("  IBTT: 'C' → vocab['C'] = 8 → embed.weight[8]")
    print("  AGTT: atom_type_0 → get_atom_type_id(0) = 8 → embed.weight[8]")
