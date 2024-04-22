from ai4bmr_core.data_models.Project import Project


def test_project_with_env():
    from dotenv import find_dotenv

    p = Project(_env_file=find_dotenv(".env"))
    assert str(p.base_dir) == "test"


def test_project_with_envprod():
    from dotenv import find_dotenv

    p = Project(_env_file=find_dotenv(".env.prod"))
    assert str(p.base_dir) == "prod"
