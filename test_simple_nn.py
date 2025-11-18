from simple_nn import train_xor


def test_xor_training_converges():
    # Train for fewer epochs to keep CI fast
    _, final_loss = train_xor(num_epochs=80, lr=0.1)

    # Just check it learns *something* reasonable
    assert final_loss < 0.2